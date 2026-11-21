#!/usr/bin/env python3
"""
Converter video GPU CUDA (NVENC) dengan GUI sederhana (tkinter).
- Pilih file video (VOB, MP4, MKV, AVI, MOV, dll) atau folder VIDEO_TS (DVD)
- Pilih format output (MP4 / MKV) dan codec (H.264 / HEVC)
- Atur kualitas, preset GPU, resize opsional
- Progress bar real-time
- Fallback otomatis ke CPU jika CUDA tidak tersedia
"""

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

VIDEO_EXTS = [".vob", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m2ts", ".mpg", ".mpeg", ".webm"]


def _resolve_exe(name):
    """Cari executable: PATH dulu, lalu lokasi umum winget untuk ffmpeg."""
    path = shutil.which(name)
    if path:
        return path
    # lokasi umum hasil 'winget install Gyan.FFmpeg'
    base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            if name + ".exe" in files:
                return os.path.join(root, name + ".exe")
    return None


FFMPEG = _resolve_exe("ffmpeg")
FFPROBE = _resolve_exe("ffprobe")


# ------------------------- logika konversi -------------------------

def run_capture(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def ffmpeg_available():
    return FFMPEG is not None


def detect_cuda():
    if not FFMPEG:
        return False
    try:
        hw = run_capture([FFMPEG, "-hide_banner", "-hwaccels"])
        enc = run_capture([FFMPEG, "-hide_banner", "-encoders"])
        out = (hw.stdout or "") + (hw.stderr or "")
        encs = (enc.stdout or "") + (enc.stderr or "")
        return "cuda" in out.lower() and "nvenc" in encs.lower()
    except Exception:
        return False


def list_vobs(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".vob")]
    main = [f for f in files
            if not re.match(r"(?i)video_ts\.vob$", f) and not re.search(r"(?i)_0\.vob$", f)]
    if not main:
        main = files
    main.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])
    return [os.path.join(folder, f) for f in main]


def get_duration(path):
    try:
        r = run_capture([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", path])
        return float(r.stdout.strip())
    except Exception:
        return None


class Converter(threading.Thread):
    """Thread konversi supaya GUI tidak freeze."""

    def __init__(self, cfg, msg_queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = msg_queue
        self.proc = None
        self._cancel = False

    def cancel(self):
        self._cancel = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _log(self, text):
        self.q.put(("log", text))

    def _progress(self, pct):
        self.q.put(("progress", pct))

    def _done(self, ok, msg):
        self.q.put(("done", (ok, msg)))

    def _build_jobs(self):
        """Kembalikan list job: (input_args, output_path, duration)."""
        c = self.cfg
        jobs = []
        if c["vob_concat"]:
            vobs = list_vobs(c["inputs"][0])
            if not vobs:
                raise RuntimeError("Tidak ada file VOB di folder tersebut.")
            jobs.append((["-i", "concat:" + "|".join(vobs)], c["output"], None))
            return jobs

        inputs = c["inputs"]
        if len(inputs) == 1:
            jobs.append((["-i", inputs[0]], c["output"], get_duration(inputs[0])))
            return jobs

        # banyak file -> output per file di folder yang dipilih
        out_dir = c["output"]  # di mode multi, 'output' = folder tujuan
        os.makedirs(out_dir, exist_ok=True)
        for f in inputs:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(out_dir, base + c["ext"])
            jobs.append((["-i", f], out, get_duration(f)))
        return jobs

    def run(self):
        c = self.cfg
        try:
            jobs = self._build_jobs()
        except Exception as e:
            self._done(False, str(e))
            return

        use_cuda = c["use_cuda"]
        if c["codec"] == "hevc":
            vcodec = "hevc_nvenc" if use_cuda else "libx265"
        else:
            vcodec = "h264_nvenc" if use_cuda else "libx264"

        total = len(jobs)
        t_start = time.time()
        for i, (input_args, out_path, duration) in enumerate(jobs, 1):
            if self._cancel:
                self._done(False, "Dibatalkan pengguna.")
                return

            self._log(f"[{i}/{total}] Mengonversi -> {os.path.basename(out_path)}")

            cmd = [FFMPEG, "-hide_banner", "-y"]
            if use_cuda:
                cmd += ["-hwaccel", "cuda"]
            cmd += input_args
            if use_cuda:
                cmd += ["-c:v", vcodec, "-preset", c["preset"], "-cq", str(c["cq"]), "-rc", "vbr"]
            else:
                cmd += ["-c:v", vcodec, "-preset", "medium", "-crf", str(c["cq"])]
            if c["resize"]:
                cmd += ["-vf", f"scale={c['resize']}"]
            cmd += ["-c:a", "aac", "-b:a", c["audio_bitrate"],
                    "-progress", "pipe:1", "-nostats", out_path]

            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace")
            except FileNotFoundError:
                self._done(False, "ffmpeg tidak ditemukan. Install dulu:\nwinget install Gyan.FFmpeg")
                return

            stderr_tail = []
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                if self._cancel:
                    break
                line = line.strip()
                if line.startswith("out_time_us=") and duration:
                    try:
                        us = int(line.split("=", 1)[1])
                        pct = min(100, int(us / 1_000_000 / duration * 100))
                        self._progress(pct)
                    except ValueError:
                        pass
            # kumpulkan stderr untuk pesan error
            if self.proc.stderr:
                try:
                    stderr_tail = self.proc.stderr.readlines()[-8:]
                except Exception:
                    pass
            self.proc.wait()

            if self._cancel:
                self._done(False, "Dibatalkan pengguna.")
                return
            if self.proc.returncode != 0:
                detail = "".join(stderr_tail).strip()
                self._done(False, f"Konversi gagal pada {os.path.basename(out_path)}.\n\n{detail}")
                return

            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            self._log(f"      Selesai ({size_mb:.1f} MB)")
            self._progress(0)

        elapsed = time.time() - t_start
        self._done(True, f"Semua konversi selesai ({total} file, {elapsed:.1f} detik).")


# ------------------------- GUI -------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CUDA Video Converter")
        self.geometry("640x620")
        self.minsize(620, 600)

        self.msg_q = queue.Queue()
        self.worker = None
        self.input_paths = []
        self.use_cuda = detect_cuda() and ffmpeg_available()

        self._build_ui()
        self._poll_queue()

    # ----- UI builders -----
    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # --- Input ---
        frm_in = ttk.LabelFrame(self, text="1. File Input")
        frm_in.pack(fill="x", **pad)

        self.lbl_input = ttk.Label(frm_in, text="(belum ada file dipilih)", foreground="gray",
                                   wraplength=580, justify="left")
        self.lbl_input.pack(fill="x", padx=8, pady=(6, 2))

        row = ttk.Frame(frm_in)
        row.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(row, text="Pilih File Video...", command=self.pick_files).pack(side="left")
        ttk.Button(row, text="Pilih Folder DVD (VIDEO_TS)...", command=self.pick_dvd_folder).pack(side="left", padx=6)
        ttk.Button(row, text="Hapus", command=self.clear_inputs).pack(side="left")

        # --- Output ---
        frm_out = ttk.LabelFrame(self, text="2. Output")
        frm_out.pack(fill="x", **pad)

        row2 = ttk.Frame(frm_out)
        row2.pack(fill="x", padx=8, pady=6)

        ttk.Label(row2, text="Format:").grid(row=0, column=0, sticky="w")
        self.var_format = tk.StringVar(value="mp4")
        ttk.Combobox(row2, textvariable=self.var_format, width=6, state="readonly",
                     values=["mp4", "mkv"]).grid(row=0, column=1, sticky="w", padx=(4, 14))

        ttk.Label(row2, text="Codec:").grid(row=0, column=2, sticky="w")
        self.var_codec = tk.StringVar(value="h264")
        ttk.Combobox(row2, textvariable=self.var_codec, width=6, state="readonly",
                     values=["h264", "hevc"]).grid(row=0, column=3, sticky="w", padx=(4, 14))

        ttk.Label(row2, text="Kualitas (CQ):").grid(row=0, column=4, sticky="w")
        self.var_cq = tk.IntVar(value=23)
        ttk.Spinbox(row2, from_=14, to=35, textvariable=self.var_cq, width=5).grid(
            row=0, column=5, sticky="w", padx=(4, 0))

        row3 = ttk.Frame(frm_out)
        row3.pack(fill="x", padx=8, pady=(0, 6))

        ttk.Label(row3, text="Preset GPU:").grid(row=0, column=0, sticky="w")
        self.var_preset = tk.StringVar(value="p5")
        ttk.Combobox(row3, textvariable=self.var_preset, width=4, state="readonly",
                     values=["p1", "p2", "p3", "p4", "p5", "p6", "p7"]).grid(row=0, column=1, sticky="w", padx=(4, 14))

        ttk.Label(row3, text="Resize (opsional):").grid(row=0, column=2, sticky="w")
        self.var_resize = tk.StringVar(value="")
        ttk.Entry(row3, textvariable=self.var_resize, width=12).grid(row=0, column=3, sticky="w", padx=(4, 4))
        ttk.Label(row3, text="mis. 1280:-2", foreground="gray").grid(row=0, column=4, sticky="w")

        ttk.Label(row3, text="Audio:").grid(row=0, column=5, sticky="w", padx=(10, 0))
        self.var_abr = tk.StringVar(value="160k")
        ttk.Combobox(row3, textvariable=self.var_abr, width=7, state="readonly",
                     values=["128k", "160k", "192k", "256k", "320k"]).grid(row=0, column=6, sticky="w", padx=(4, 0))

        # --- Status engine ---
        engine = "GPU NVIDIA (NVENC)" if self.use_cuda else "CPU (CUDA tidak terdeteksi)"
        self.lbl_engine = ttk.Label(self, text=f"Engine: {engine}",
                                    foreground=("green" if self.use_cuda else "orange"))
        self.lbl_engine.pack(anchor="w", padx=12)

        # --- Progress ---
        frm_prog = ttk.LabelFrame(self, text="3. Proses")
        frm_prog.pack(fill="both", expand=True, **pad)

        self.progress = ttk.Progressbar(frm_prog, maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(6, 2))

        self.lbl_status = ttk.Label(frm_prog, text="Siap.", foreground="gray")
        self.lbl_status.pack(anchor="w", padx=8)

        self.txt_log = tk.Text(frm_prog, height=10, state="disabled", wrap="word",
                               font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=6)

        # --- Tombol aksi ---
        row_btn = ttk.Frame(self)
        row_btn.pack(fill="x", padx=10, pady=(0, 10))
        self.btn_start = ttk.Button(row_btn, text="Mulai Konversi", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(row_btn, text="Batal", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)

    # ----- UI actions -----
    def pick_files(self):
        types = [("File Video", " ".join("*" + e for e in VIDEO_EXTS)), ("Semua File", "*.*")]
        paths = filedialog.askopenfilenames(title="Pilih file video", filetypes=types)
        if paths:
            self.input_paths = list(paths)
            self._show_inputs()

    def pick_dvd_folder(self):
        folder = filedialog.askdirectory(title="Pilih folder VIDEO_TS / DVD")
        if folder:
            self.input_paths = [folder]
            self._show_inputs(dvd=True)

    def clear_inputs(self):
        self.input_paths = []
        self.lbl_input.config(text="(belum ada file dipilih)", foreground="gray")

    def _show_inputs(self, dvd=False):
        n = len(self.input_paths)
        if dvd:
            txt = f"[DVD] {self.input_paths[0]}"
        elif n == 1:
            txt = self.input_paths[0]
        else:
            txt = f"{n} file dipilih: " + ", ".join(os.path.basename(p) for p in self.input_paths[:4]) + (" ..." if n > 4 else "")
        self.lbl_input.config(text=txt, foreground="black")

    def _log(self, text):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    def start(self):
        if not self.input_paths:
            messagebox.showwarning("Belum ada input", "Pilih file video atau folder DVD dulu.")
            return
        if not ffmpeg_available():
            messagebox.showerror("FFmpeg tidak ada", "ffmpeg tidak ditemukan.\nInstall: winget install Gyan.FFmpeg")
            return

        resize = self.var_resize.get().strip() or None
        if resize and not re.match(r"^\d+:(\d+|-1|-2)$", resize):
            messagebox.showwarning("Resize salah", "Format resize mis. 1280:720 atau 1280:-2")
            return

        ext = "." + self.var_format.get()
        is_dvd = len(self.input_paths) == 1 and os.path.isdir(self.input_paths[0])
        multi = len(self.input_paths) > 1

        # tentukan output
        if is_dvd:
            default = os.path.join(os.path.dirname(self.input_paths[0].rstrip("\\/")), "output" + ext)
            out = filedialog.asksaveasfilename(defaultextension=ext, initialfile=os.path.basename(default),
                                               filetypes=[(ext.upper(), "*" + ext)])
            if not out:
                return
            output, vob_concat = out, True
        elif multi:
            out_dir = filedialog.askdirectory(title="Pilih folder tujuan output")
            if not out_dir:
                return
            output, vob_concat = out_dir, False
        else:
            src = self.input_paths[0]
            default = os.path.splitext(src)[0] + ext
            out = filedialog.asksaveasfilename(defaultextension=ext, initialfile=os.path.basename(default),
                                               filetypes=[(ext.upper(), "*" + ext)])
            if not out:
                return
            output, vob_concat = out, False

        cfg = {
            "inputs": self.input_paths,
            "output": output,
            "ext": ext,
            "codec": self.var_codec.get(),
            "cq": self.var_cq.get(),
            "preset": self.var_preset.get(),
            "resize": resize,
            "audio_bitrate": self.var_abr.get(),
            "vob_concat": vob_concat,
            "use_cuda": self.use_cuda,
        }

        self.txt_log.config(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.config(state="disabled")
        self.progress["value"] = 0
        self.lbl_status.config(text="Memproses...", foreground="black")
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")

        self.worker = Converter(cfg, self.msg_q)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
        self.lbl_status.config(text="Membatalkan...", foreground="orange")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                    self.lbl_status.config(text=f"Proses: {payload}%")
                elif kind == "done":
                    ok, msg = payload
                    self._log(msg)
                    self.lbl_status.config(text=("Selesai." if ok else "Gagal."),
                                           foreground=("green" if ok else "red"))
                    self.btn_start.config(state="normal")
                    self.btn_cancel.config(state="disabled")
                    if ok:
                        messagebox.showinfo("Selesai", msg)
                    else:
                        messagebox.showerror("Gagal", msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


if __name__ == "__main__":
    App().mainloop()
