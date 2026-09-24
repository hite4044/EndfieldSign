import argparse
import os
from dataclasses import dataclass
from io import BytesIO
from os import makedirs
from os.path import basename
from subprocess import Popen, PIPE, DEVNULL
from time import perf_counter, time

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

raw_print = print
output_log = open(f"output_{round(time())}.log", "w")


def print(*args, **kwargs):
    raw_print(*args, **kwargs)
    raw_print(*args, **({**kwargs, **{"file": output_log}}))


def color_similarity(rgb1: tuple, rgb2: tuple, brightness_weight: float = 3.0) -> float:
    """
    计算两个 RGB 颜色 (0~255) 的相似度，输出 0.0 ~ 1.0。
    亮度差异在结果中占更大比重。
    """
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2

    # ---- 亮度分量（感知亮度，人眼对绿色最敏感）----
    lum1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
    lum2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
    lum_diff = abs(lum1 - lum2) / 255.0

    # ---- 色度分量（RGB 各通道差异的欧氏距离）----
    chroma_diff = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    chroma_diff /= (255.0 * (3 ** 0.5))

    # ---- 加权融合 ----
    total_weight = 1.0 + brightness_weight
    diff = (chroma_diff + brightness_weight * lum_diff) / total_weight
    return max(0.0, 1.0 - diff)


class VideoDecodeError(Exception):
    pass


class VideoDecoder:
    def __init__(self, video_file: str, hwaccel: str = "cuda", detect_fps: float = 0.5):
        self.video_file = video_file
        # 根据参数拼接硬件加速选项，none/空 时不使用
        hwaccel_opt = "" if hwaccel in (None, "", "none") else f"-hwaccel {hwaccel}"
        self.process = Popen(
            " ".join(
                ["ffmpeg", '-nostdin', hwaccel_opt,
                 f'-i "{video_file}" -vf "fps={detect_fps}"',
                 '-f image2pipe -c:v mjpeg -q:v 10 -']
            ),
            stdout=PIPE,
            stderr=DEVNULL,
        )

    def jpeg_stream_generator(self):
        """
        从 proc.stdout 中读取连续的 JPEG 数据流，
        每次迭代返回一个包含单张 JPEG 的 BytesIO 对象。
        """
        buffer = b''
        while True:
            frame_buffer = b''
            while True:
                new_content = self.process.stdout.read(4096)
                if len(new_content) == 0:
                    break
                if b"\xFF\xD9" in new_content:
                    buffer += new_content
                    stop_pos = buffer.rfind(b"\xFF\xD9")
                    frame_buffer = buffer[:stop_pos + 2]
                    buffer = buffer[stop_pos + 2:]
                    break
                elif buffer.endswith(b"\xFF") and new_content.startswith(b"\xD9"):
                    buffer += new_content
                    frame_buffer = buffer[:-len(new_content) + 1]
                    buffer = buffer[-len(new_content) + 1:]
                    break
                else:
                    buffer += new_content
            if frame_buffer == b'' and buffer == b'':
                break
            yield BytesIO(frame_buffer)

    def frames(self):
        frame_count = 0
        for frame_io in self.jpeg_stream_generator():
            frame_count += 1
            try:
                yield Image.open(frame_io)
            except UnidentifiedImageError:
                print(f"[{basename(self.video_file)}] Frame {frame_count} is corrupted")
                with open(f"{basename(self.video_file).split('.')[0]}_error_{frame_count}.jpg", "wb") as f:
                    f.write(frame_io.getbuffer())
                yield Image.new("RGB", (1920, 1080))


class VideoCliper:
    """视频剪切器, 以复制流切割视频"""

    def __init__(self, input_video: str, from_: int, to: int, clip_output: str):
        self.proc = Popen(
            " ".join(
                ["ffmpeg", '-nostdin', f'-ss {self.fmt_time(from_)} -to {self.fmt_time(to)}',
                 f'-i "{input_video}"', '-c copy', f'"{clip_output}"']
            ),
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        self.proc.wait()
        if self.proc.returncode != 0:
            print(f"[{basename(input_video)}] Failed to clip video")

    @staticmethod
    def fmt_time(seconds: int) -> str:
        """将秒数转为 HH:MM:SS 格式"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds %= 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.000"


@dataclass
class DetectPoint:
    pos: tuple[float, float]
    color: tuple[int, int, int]
    threshold: float = 0.95


POINTS = [
    DetectPoint(pos=(0.25, 0.52), color=(255, 253, 253)),
    DetectPoint(pos=(0.5, 0.52), color=(232, 230, 230)),
    DetectPoint(pos=(0.75, 0.52), color=(255, 253, 253)),
]

FPS_FAC = 0.1


def process_file(video_file: str, detect_fps: float, temp_img_path: str,
                 temp_threshold: float, seg_split_time: int, video_pad: int,
                 hwaccel: str, endfix: str):
    print(f"[{basename(video_file)}] Start processing")
    temp_img = cv2.imread(temp_img_path)
    if temp_img is None:
        print(f"[{basename(video_file)}] Failed to load template image: {temp_img_path}")
        return

    segs = []
    crt_seg = []
    try:
        dec = VideoDecoder(video_file, hwaccel=hwaccel, detect_fps=detect_fps)
    except VideoDecodeError as e:
        print(f"[{basename(video_file)}] VideoDecodeError: {e}")
        return

    flag = True
    frame_time = 0.1
    start, stop = perf_counter(), perf_counter()
    last_report = perf_counter()
    has_passed = False

    for frame_count, frame in enumerate(dec.frames()):
        stop = perf_counter()
        frame_time = frame_time * (1 - FPS_FAC) + (stop - start) * FPS_FAC
        start = perf_counter()

        if perf_counter() - last_report > 1:
            last_report = perf_counter()
            print(f"\r[{basename(video_file)}] Processing: {frame_count},",
                  f"speed={round(1 / frame_time / detect_fps, 2)}x",
                  end=" " * 17 if not has_passed else ", SIM TEST PASSED",
                  )
            has_passed = False

        # 像素检测
        for point in POINTS:
            pixel = frame.getpixel((int(point.pos[0] * frame.width), int(point.pos[1] * frame.height)))
            similarity = color_similarity(pixel, point.color)
            if similarity < point.threshold:
                flag = True
                break
        else:
            flag = False

        if flag:
            continue
        has_passed = True

        # 模板匹配
        frame_array = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        res = cv2.matchTemplate(frame_array, temp_img, cv2.TM_SQDIFF_NORMED)

        if np.min(res) < temp_threshold:
            print(f"\n[{basename(video_file)}] [{frame_count}] Match found")
            if len(crt_seg) >= 1 and frame_count - crt_seg[-1] > seg_split_time * detect_fps:
                # 新片段开始时保存旧片段
                segs.append(crt_seg)
                crt_seg = []
            crt_seg.append(frame_count)

    if len(crt_seg) > 0:
        segs.append(crt_seg)
    print()

    # 导出片段
    for seg in segs:
        start_framecount, end_framecount = seg[0], seg[-1]
        start_sec = int(max(start_framecount / detect_fps - video_pad, 0))
        end_sec = int(end_framecount / detect_fps + video_pad)
        print(f"[{basename(video_file)}] Extract Segment: [{start_sec} ~ {end_sec}]")
        out_name = basename(video_file)
        if out_name.endswith(endfix):
            out_name = out_name[:-len(endfix)]
        VideoCliper(video_file, start_sec, end_sec, fr"segs\{out_name}_{start_sec}-{end_sec}{endfix}")


def process_dir(root, detect_fps, temp_img_path, temp_threshold,
                seg_split_time, video_pad, hwaccel, endfix):
    for name in os.listdir(root):
        if name.endswith(endfix):
            fp = os.path.join(root, name)
            try:
                process_file(fp, detect_fps, temp_img_path, temp_threshold,
                             seg_split_time, video_pad, hwaccel, endfix)
            except Exception as e:
                print(f"[{name}] SERIOUS FUCKING ERROR: {e}")
            print("================")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="EndfieldSign",
        description="从多个视频中批量检测终末地更改签名界面并且提取相关片段",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*", default=[r"D:\Desktop\Endfield"],
        help="待处理的目录或视频文件路径（可多个，不递归）；不传则使用默认目录",
    )
    parser.add_argument(
        "--hwaccel", default="cuda", metavar="DECODER",
        help="ffmpeg 硬件加速解码器，如 cuda / qsv / d3d11va；传 none 禁用（CPU解码）",
    )
    parser.add_argument(
        "--temp-threshold", type=float, default=0.05, metavar="T",
        help="模板匹配阈值（TM_SQDIFF_NORMED 差值，越小越严格）",
    )
    parser.add_argument(
        "--detect-fps", type=float, default=0.5,
        help="检测更改签名界面的帧率",
    )
    parser.add_argument(
        "--temp-img", default="temp.png",
        help='"更改签名"文字的模板图片路径',
    )
    parser.add_argument(
        "--endfix", default=".mkv",
        help="视频文件后缀",
    )
    parser.add_argument(
        "--seg-split-time", type=int, default=60,
        help='判定为新片段的秒数，检测到"更改签名"的帧之间超过该秒数会被切割',
    )
    parser.add_argument(
        "--video-pad", type=int, default=10,
        help="分割片段前后的预留秒数",
    )
    return parser


def main():
    args = build_parser().parse_args()

    makedirs("segs", exist_ok=True)

    paths = args.paths if args.paths else [r"D:\Desktop\Endfield"]
    for path in paths:
        if os.path.isdir(path):
            process_dir(path, args.detect_fps, args.temp_img, args.temp_threshold,
                        args.seg_split_time, args.video_pad, args.hwaccel, args.endfix)
        elif os.path.isfile(path):
            try:
                process_file(path, args.detect_fps, args.temp_img, args.temp_threshold,
                             args.seg_split_time, args.video_pad, args.hwaccel, args.endfix)
            except Exception as e:
                print(f"[{basename(path)}] SERIOUS FUCKING ERROR: {e}")
        else:
            print(f"Path not found: {path}")


if __name__ == "__main__":
    main()
