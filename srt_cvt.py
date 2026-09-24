# -*- coding: utf-8 -*-
"""
根据 EDL 时间线索引生成 SRT 字幕：
每个片段在 [录制入点, 录制出点] 期间显示静态时间 = DATE TIME1 + START_SEC + 10s
"""
import csv
import re
from datetime import datetime, timedelta

CSV_FILE   = "tl.csv"
SRT_FILE   = "Timeline 1.srt"
OFFSET_SEC = 10  # TIME1 + START_SEC + 10sec

# 匹配素材名：任意非数字前缀 + 日期 + 时分秒 + 起始秒-结束秒
NAME_RE = re.compile(
    r'[^\d]*(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})_(-?\d+)-(-?\d+)'
)

def timecode_to_seconds(tc: str, fps: float) -> float:
    """HH:MM:SS:FF -> 秒"""
    h, m, s, f = map(int, tc.split(":"))
    return h * 3600 + m * 60 + s + f / fps

def srt_time(seconds: float) -> str:
    """秒 -> SRT 时间格式 HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

entries = []
with open(CSV_FILE, encoding="utf-8-sig") as fp:
    for row in csv.reader(fp):
        if len(row) < 12:
            continue
        seq = row[0].strip()
        if not re.fullmatch(r"\d+", seq):        # 跳过表头 / M2 标记行
            continue

        name    = row[11].strip()
        rec_in  = row[9].strip()                 # 录制入点
        rec_out = row[10].strip()                # 录制出点
        fps     = float(row[17]) if row[17].strip() else 30.0

        m = NAME_RE.search(name)
        if not m or not rec_in or not rec_out:
            continue

        date_str = m.group(1)
        hh, mm, ss = m.group(2), m.group(3), m.group(4)
        start_sec = int(m.group(5))              # START_SEC，可为负

        # 静态时间：DATE TIME1 + START_SEC + 10s
        base = datetime.strptime(f"{date_str} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
        static_time = base + timedelta(seconds=start_sec + OFFSET_SEC)
        text = static_time.strftime("%Y-%m-%d %H:%M:%S")

        # 字幕起止：录制入点 -> 录制出点
        t_in  = timecode_to_seconds(rec_in, fps)
        t_out = timecode_to_seconds(rec_out, fps)

        entries.append((t_in, t_out, text))

entries.sort(key=lambda e: e[0])

with open(SRT_FILE, "w", encoding="utf-8") as fp:
    for i, (t_in, t_out, text) in enumerate(entries, 1):
        fp.write(f"{i}\n{srt_time(t_in)} --> {srt_time(t_out)}\n{text}\n\n")

print(f"完成：共生成 {len(entries)} 条字幕 -> {SRT_FILE}")
