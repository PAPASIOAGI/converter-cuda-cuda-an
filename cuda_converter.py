#!/usr/bin/env python3
"""
Converter video menggunakan GPU NVIDIA (CUDA / NVENC).
Mendukung VOB (termasuk penggabungan folder VIDEO_TS DVD), MP4, MKV, AVI, MOV, TS, dll.
Otomatis fallback ke CPU (libx264/libx265) jika CUDA tidak tersedia.
"""

import argparse
import os
import re
import subprocess
import sys
import time

VIDEO_EXTS = {".vob", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m2ts", ".mpg", ".mpeg", ".webm"}

PRESET_NVENC = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]  # p1=tercepat, p7=kualitas terbaik


def run(cmd, capture=False):
    return subprocess.run(cmd, capture_output=capture, text=True, encoding="utf-8", errors="replace")


def check_ffmpeg():
    r = run(["ffmpeg", "-hide_banner", "-version"], capture=True)
    if r.returncode != 0:
        sys.exit("ERROR: ffmpeg tidak ditemukan. Install dulu (mis. winget install Gyan.FFmpeg).")
    return True


def detect_cuda():
    """Kembalikan True jika ffmpeg punya dukungan cuda + encoder nvenc."""
    hw = run(["ffmpeg", "-hide_banner", "-hwaccels"], capture=True)
    enc = run(["ffmpeg", "-hide_banner", "-encoders"], capture=True)
    out = (hw.stdout or "") + (hw.stderr or "")
    encs = (enc.stdout or "") + (enc.stderr or "")
    return "cuda" in out.lower() and "nvenc" in encs.lower()


def list_vobs(folder):
    """Ambil daftar VOB utama (bukan menu) dari folder VIDEO_TS, diurutkan alami."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(".vob")]
    # abaikan VOB menu (VIDEO_TS.VOB, VTS_xx_0.VOB berukuran menu biasanya _0)
    main = [f for f in files if not re.match(r"(?i)video_ts\.vob$", f) and not re.search(r"(?i)_0\.vob$", f)]
    if not main:
        main = files
    main.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])
    return [os.path.join(folder, f) for f in main]


def build_inputs(inputs, vob_concat):
    """Bangun daftar argumen input ffmpeg. Return (input_args, display_name, is_concat)."""
    if vob_concat:
        folder = inputs[0]
        if not os.path.isdir(folder):
            sys.exit(f"ERROR: folder tidak ditemukan: {folder}")
        vobs = list_vobs(folder)
        if not vobs:
            sys.exit(f"ERROR: tidak ada file VOB di {folder}")
        concat_str = "concat:" + "|".join(vobs)
        return ["-i", concat_str], f"{os.path.basename(os.path.normpath(folder))} ({len(vobs)} VOB)", True

    for f in inputs:
        if not os.path.isfile(f):
            sys.exit(f"ERROR: file tidak ditemukan: {f}")
    if len(inputs) == 1:
        return ["-i", inputs[0]], os.path.basename(inputs[0]), False
    # gabung banyak file dengan concat demuxer (harus codec sejenis)
    import tempfile
    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    for f in inputs:
        lst.write(f"file '{os.path.abspath(f)}'\n")
    lst.close()
    return ["-f", "concat", "-safe", "0", "-i", lst.name], f"{len(inputs)} file", True


def get_duration(input_args):
    """Ambil durasi (detik) via ffprobe untuk progress."""
    try:
        idx = input_args.index("-i") + 1
        src = input_args[idx]
        if src.startswith("concat:"):
            return None
        r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", src], capture=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def convert(inputs, output, codec, cq, preset, resolution, audio_bitrate, vob_concat, overwrite):
    use_cuda = detect_cuda()
    input_args, disp, _ = build_inputs(inputs, vob_concat)

    if codec == "hevc":
        vcodec_gpu, vcodec_cpu = "hevc_nvenc", "libx265"
    else:
        vcodec_gpu, vcodec_cpu = "h264_nvenc", "libx264"

    cmd = ["ffmpeg", "-hide_banner", "-y" if overwrite else "-n"]
    if use_cuda:
        cmd += ["-hwaccel", "cuda"]
    cmd += input_args

    if use_cuda:
        cmd += ["-c:v", vcodec_gpu, "-preset", preset, "-cq", str(cq), "-rc", "vbr"]
    else:
        cpu_preset = "medium"
        cmd += ["-c:v", vcodec_cpu, "-preset", cpu_preset, "-crf", str(cq)]

    if resolution:
        cmd += ["-vf", f"scale={resolution}"]
    cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-progress", "pipe:1", "-nostats", output]

    mode = f"GPU ({vcodec_gpu})" if use_cuda else f"CPU ({vcodec_cpu}) - CUDA tidak tersedia"
    duration = get_duration(input_args)

    print(f"Input   : {disp}")
    print(f"Output  : {output}")
    print(f"Encoder : {mode} | preset={preset if use_cuda else 'medium'} | cq/crf={cq}")
    if resolution:
        print(f"Resize  : {resolution}")
    print("-" * 60)

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace")
    last_pct = -1
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us=") and duration:
            try:
                us = int(line.split("=", 1)[1])
                pct = min(100, int(us / 1_000_000 / duration * 100))
                if pct != last_pct:
                    last_pct = pct
                    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                    print(f"\r[{bar}] {pct:3d}%", end="", flush=True)
            except ValueError:
                pass
    proc.wait()
    elapsed = time.time() - t0
    print()
    if proc.returncode != 0:
        print("ERROR: konversi gagal. Jalankan ulang tanpa --progress untuk detail (atau cek parameter).")
        sys.exit(1)
    size = os.path.getsize(output) / (1024 * 1024)
    print(f"Selesai dalam {elapsed:.1f} detik. Ukuran output: {size:.1f} MB -> {output}")


def main():
    p = argparse.ArgumentParser(
        description="Converter video GPU CUDA (NVENC). Mendukung VOB/MP4/MKV/AVI/dll.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Contoh:
  python cuda_converter.py input.vob -o output.mp4
  python cuda_converter.py "D:\\VIDEO_TS" --vob-concat -o film.mp4
  python cuda_converter.py *.mkv -o hasil.mp4 -c hevc --cq 26
  python cuda_converter.py in.avi -o out.mp4 --resize 1280:-2 --preset p7
""")
    p.add_argument("inputs", nargs="+", help="File input, atau folder VIDEO_TS jika --vob-concat")
    p.add_argument("-o", "--output", required=True, help="File output (mis. output.mp4)")
    p.add_argument("-c", "--codec", choices=["h264", "hevc"], default="h264",
                   help="Codec video output (default: h264)")
    p.add_argument("--cq", type=int, default=23,
                   help="Kualitas (CQ utk GPU / CRF utk CPU), makin kecil makin bagus (default: 23)")
    p.add_argument("--preset", choices=PRESET_NVENC, default="p5",
                   help="Preset NVENC p1(cepat)..p7(kualitas), default p5")
    p.add_argument("--resize", metavar="WxH", default=None,
                   help="Resize, mis. 1280:720 atau 1280:-2 (auto rasio)")
    p.add_argument("--audio-bitrate", default="160k", help="Bitrate audio AAC (default: 160k)")
    p.add_argument("--vob-concat", action="store_true",
                   help="Gabungkan semua VOB utama di folder (untuk DVD VIDEO_TS)")
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                   help="Jangan timpa file output yang sudah ada")
    args = p.parse_args()

    check_ffmpeg()
    convert(args.inputs, args.output, args.codec, args.cq, args.preset,
            args.resize, args.audio_bitrate, args.vob_concat, args.overwrite)


if __name__ == "__main__":
    main()
