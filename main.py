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

    :param rgb1: (r, g, b) 元组
    :param rgb2: (r, g, b) 元组
    :param brightness_weight: 亮度差异的放大系数，默认 3.0（越大亮度影响越强）
    :return: 0.0（完全不同）~ 1.0（完全相同）
    """
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2

    # ---- 亮度分量（感知亮度，人眼对绿色最敏感）----
    lum1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
    lum2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2

    # 亮度差异归一化到 0~1
    lum_diff = abs(lum1 - lum2) / 255.0

    # ---- 色度分量（RGB 各通道差异的欧氏距离）----
    chroma_diff = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    chroma_diff /= (255.0 * (3 ** 0.5))  # 归一化到 0~1（理论最大值 255*√3）

    # ---- 加权融合 ----
    total_weight = 1.0 + brightness_weight
    diff = (chroma_diff + brightness_weight * lum_diff) / total_weight

    return max(0.0, 1.0 - diff)


class VideoDecodeError(Exception):
    pass


class VideoDecoder:
    RES_PATTERN = r"\d+x\d+"

    def __init__(self, video_file: str):
        self.video_file = video_file
        self.process = Popen(
            " ".join(
                ["ffmpeg",
                 '-nostdin',
                 '-hwaccel cuda', # 使用CUDA加速
                 f'-i "{video_file}" -vf "fps={DETECT_FPS}"',
                 '-f image2pipe -c:v mjpeg -q:v 10 -'
                 ]
            ),
            stdout=PIPE,
            stderr=DEVNULL,
        )
        # self.resolution = (1920, 1080)
        # while True:
        #     li = self.process.stderr.readline()
        #     if b"Stream #0:0: Video" in li:
        #         text = li.decode("utf-8")
        #         resolution_text = re.search(self.RES_PATTERN, text)
        #         if resolution_text is None:
        #             raise VideoDecodeError(f"Could not find video resolution in [{text}]")
        #         self.resolution = tuple(map(int, resolution_text.group().split("x")))
        #         break
        #     if li == b'':
        #         raise VideoDecodeError("Video resolution miss")
        #     if self.process.poll() is not None:
        #         raise VideoDecodeError("Decoding process unexpectedly quit early")
        # self.process.stderr.close()

    def jpeg_stream_generator(self):
        """
        从 proc.stdout 中读取连续的 JPEG 数据流，
        每次迭代返回一个包含单张 JPEG 的 BytesIO 对象。
        """
        buffer = b''
        while True:
            frame_buffer = b''
            while True:
                # 读取新内容，没有内容就退出
                new_content = self.process.stdout.read(4096)
                if len(new_content) == 0:
                    break

                # 如果新的内容找到了结束标记，就说明新内容包含了JPEG的结尾
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

            # buffer吃完了，frame_buffer也是空的，已经结束了
            if frame_buffer == b'' and buffer == b'':
                break

            # print(f"Frame buffer size: {len(frame_buffer)}")
            yield BytesIO(frame_buffer)

    def frames(self):
        frame_count = 0
        for frame_io in self.jpeg_stream_generator():
            frame_count += 1
            # print(f"Frame {frame_count}")
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
                ["ffmpeg",
                 '-nostdin',
                 f'-ss {self.fmt_time(from_)} -to {self.fmt_time(to)}',
                 f'-i "{input_video}"',
                 '-c copy',
                 f'"{clip_output}"'
                 ]
            ),
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        self.proc.wait()
        if self.proc.returncode != 0:
            print(f"[{basename(input_video)}] Failed to clip video")

    @staticmethod
    def fmt_time(seconds: int) -> str:
        """
        将秒数转为 HH:MM:SS 格式
        """
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

temp_threshold = 0.05
FPS_FAC = 0.1


def process_file(video_file: str):
    print(f"[{basename(video_file)}] Start processing")
    temp_img = cv2.imread(TEMP_IMG)  # 读取模板图片

    # 获取视频修改签名片段
    segs = []
    crt_seg = []

    try:
        dec = VideoDecoder(video_file)
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
        frame_time = frame_time * (1 - FPS_FAC) + (stop - start) * FPS_FAC  # 加权平均值计算帧时间
        start = perf_counter()

        # 像素检测
        if perf_counter() - last_report > 1:
            last_report = perf_counter()
            print(f"\r[{basename(video_file)}] Processing: {frame_count},",
                  f"speed={round(1 / frame_time / DETECT_FPS, 2)}x",
                  end=" " * 17 if not has_passed else ", SIM TEST PASSED", )
            has_passed = False
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
        # print(f"[{basename(video_file)}] [{frame_count}] Sim test passed")

        # 模板匹配
        frame_array = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        res = cv2.matchTemplate(frame_array, temp_img, cv2.TM_SQDIFF_NORMED)  # 第二种方式匹配
        if np.min(res) < temp_threshold:
            print(f"\n[{basename(video_file)}] [{frame_count}] Match found")
            if len(crt_seg) >= 1 and frame_count - crt_seg[-1] > SEG_SPLIT_TIME:  # 新片段开始时保存旧片段
                segs.append(crt_seg)
                crt_seg = []
            crt_seg.append(frame_count)
            # frame.save(f"images\\{frame_count}.png")

    if len(crt_seg) > 0:  # 处理剩余片段
        segs.append(crt_seg)
    print()

    # 导出片段
    for seg in segs:
        start_framecount, end_framecount = seg[0], seg[-1]
        start_sec = int(max(start_framecount / DETECT_FPS - VIDEO_PAD, 0))
        end_sec = int(end_framecount / DETECT_FPS + VIDEO_PAD)
        print(f"[{basename(video_file)}] Extract Segment: [{start_sec} ~ {end_sec}]")
        VideoCliper(video_file, start_sec, end_sec,
                    fr"segs\{basename(video_file).replace('.mkv', '')}_{start_sec}-{end_sec}.mkv")


def process_dir(root):
    for name in os.listdir(root):
        if name.endswith(PROCESS_ENDFIX):
            fp = os.path.join(root, name)
            try:
                process_file(fp)
            except Exception as e:
                print(f"[{name}] SERIOUS FUCKING ERROR: {e}")
            print("================")


makedirs("segs", exist_ok=True)
"""
EndfieldSign
从多个视频中批量检测终末地更改签名界面并且提取相关片段

需要预先下载带CUDA解码器的ffmpeg二进制exe并添加至环境变量。
如果需要使用其他解码器，请修改66行的硬件加速参数。
默认使用NVDIA硬件解码器，如果换成CPU解码要删掉66行。

使用前先转到更改签名界面，使用截图软件框选"更改签名"四个字，保存png到项目根目录下。
再指定 TEMP_IMG 至保存的图片，再根据实际情况修改 PROCESS_ENDFIX 和 paths

然后直接启动，程序会忽略异常的ffmpeg解码进程退出。
处理完成的片段在segs文件夹，并在文件名结尾标示了截取片段范围。
"""
DETECT_FPS = 0.5  # 检测更改签名界面的帧率
TEMP_IMG = "temp.png"  # "更改签名"文字的图片（游戏内区域截图）
PROCESS_ENDFIX = ".mkv"  # 视频文件后缀

SEG_SPLIT_TIME = 60  # 是否判定为新片段的秒数，检测到"更改签名"的帧之间如果超过了这个秒数就会被切割。
VIDEO_PAD = 10  # 分割片段前后的预留秒数

paths = [
    r"D:\Desktop\Endfield",
]  # 待处理的目录，只遍历指定目录下的文件，不递归
for dir_path in paths:
    process_dir(dir_path)

# process_file(r"E:\游戏录屏\终末地\1.4\2026.8\Archived.实况 2026-08-28 10-29-29.mkv")
