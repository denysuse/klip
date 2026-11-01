#!/usr/bin/env python3
"""
klip.py - Potong video panjang jadi klip Shorts + peta timestamp penting.

Alur:
  1. Ekstrak audio, transkrip lokal dengan faster-whisper (timestamp per kata).
  2. Susun jendela kandidat dari batas segmen bicara, beri skor
     (kepadatan bicara, kata pemicu/hook, jeda alami di awal-akhir, keyakinan model).
  3. Pilih N klip terbaik yang tidak tumpang tindih.
  4. Potong dengan ffmpeg (default vertikal 9:16 latar blur untuk Shorts),
     buat SRT per klip, dan tulis laporan timestamp (Markdown + JSON).

Contoh:
  python3 klip.py video-panjang.mp4 --jumlah 5
  python3 klip.py podcast.mp4 --jumlah 3 --min-dur 25 --max-dur 55 --lang id
  python3 klip.py video.mp4 --format asli --caption

Butuh: ffmpeg, ffprobe, paket faster-whisper (lihat venv di folder ini).
Tidak ada data yang diunggah; semua proses lokal.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

# Kata pemicu yang biasanya menandai momen layak klip (Indonesia + Inggris).
HOOK_WORDS_ID = [
    "penting", "rahasia", "cara", "kenapa", "mengapa", "jangan", "kesalahan",
    "tips", "trik", "masalah", "solusi", "bukti", "ternyata", "faktanya",
    "perhatikan", "bayangkan", "contohnya", "hasilnya", "mulai", "berhenti",
    "selalu", "tidak pernah", "paling", "pertama", "terakhir", "gratis",
    "ingat", "dengarkan", "lihat", "coba", "harus", "wajib", "bahaya",
    "rugi", "untung", "jutaan", "miliar", "persen",
]
HOOK_WORDS_EN = [
    "important", "secret", "how", "why", "never", "always", "mistake",
    "problem", "solution", "actually", "truth", "listen", "watch", "imagine",
    "example", "result", "first", "last", "best", "worst", "free", "money",
    "stop", "start", "must", "million", "percent",
]
FILLER = {"eh", "eee", "anu", "um", "uh", "hmm", "kayak", "gitu", "kan", "ya", " sih"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def fmt_ts(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def fmt_srt(sec):
    return fmt_ts(sec).replace(".", ",")


def probe_duration(path):
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
    return float(json.loads(p.stdout)["format"]["duration"])


def probe_size(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", path])
    st = json.loads(p.stdout)["streams"][0]
    return int(st["width"]), int(st["height"])


def extract_audio(video, wav):
    p = run(["ffmpeg", "-y", "-v", "error", "-i", video,
             "-vn", "-ac", "1", "-ar", "16000", wav])
    if p.returncode != 0:
        raise RuntimeError("ffmpeg ekstrak audio gagal: " + p.stderr[-400:])


def load_wav_numpy(wav):
    """Baca WAV PCM 16-bit mono jadi array float32 (hindari decodasi PyAV)."""
    import wave
    import numpy as np
    with wave.open(wav, "rb") as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def transcribe(wav, model_name, lang):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    kwargs = dict(word_timestamps=True, vad_filter=True,
                  vad_parameters=dict(min_silence_duration_ms=500))
    if lang and lang != "auto":
        kwargs["language"] = lang
    audio, _sr = load_wav_numpy(wav)
    segments, info = model.transcribe(audio, **kwargs)
    segs = []
    for s in segments:
        words = [{"w": w.word.strip(), "s": w.start, "e": w.end}
                 for w in (s.words or []) if w.word.strip()]
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                     "words": words, "conf": getattr(s, "avg_logprob", -0.5)})
    return segs, info.language


def score_window(win_words, text, duration, gap_before, ends_sentence, conf):
    """Skor heuristik 0-100 untuk satu jendela kandidat."""
    n = len(win_words)
    if n == 0 or duration <= 0:
        return 0.0
    wps = n / duration  # kata per detik
    score = 0.0
    # Kepadatan bicara ideal 2.0-3.2 kata/detik (bicara normal-penuh isi)
    if 2.0 <= wps <= 3.2:
        score += 30
    elif 1.4 <= wps < 2.0 or 3.2 < wps <= 3.8:
        score += 18
    else:
        score += 6
    low = " " + text.lower() + " "
    hooks = sum(1 for w in HOOK_WORDS_ID + HOOK_WORDS_EN
                if re.search(r"[\s\W]" + re.escape(w) + r"[\s\W]", low))
    score += min(25, hooks * 5)
    if "?" in text:
        score += 8  # pertanyaan biasanya hook kuat
    if re.search(r"\d", text):
        score += 5  # angka spesifik menarik perhatian
    if gap_before >= 0.6:
        score += 10  # mulai setelah jeda = awal yang bersih
    if ends_sentence:
        score += 10  # berakhir di ujung kalimat
    score += max(-8, min(12, (conf + 0.5) * 16))  # keyakinan transkrip
    fillers = sum(1 for w in win_words if w["w"].lower() in FILLER)
    score -= min(8, fillers)
    return round(max(0.0, min(100.0, score)), 1)


def build_candidates(segs, min_dur, max_dur):
    cands = []
    n = len(segs)
    for i in range(n):
        start = segs[i]["start"]
        gap_before = start - segs[i - 1]["end"] if i > 0 else 999.0
        words, texts, confs = [], [], []
        for j in range(i, n):
            dur = segs[j]["end"] - start
            if dur > max_dur:
                break
            words.extend(segs[j]["words"])
            texts.append(segs[j]["text"])
            confs.append(segs[j]["conf"])
            if dur >= min_dur:
                text = " ".join(texts)
                ends = bool(re.search(r"[.!?…]\s*$", segs[j]["text"]))
                conf = sum(confs) / len(confs)
                cands.append({
                    "start": round(start, 2), "end": round(segs[j]["end"], 2),
                    "duration": round(dur, 2), "text": text,
                    "score": score_window(words, text, dur, gap_before, ends, conf),
                    "_words": words,
                })
    cands.sort(key=lambda c: -c["score"])
    return cands


def pick_top(cands, jumlah, min_gap=5.0):
    picked = []
    for c in cands:
        ok = True
        for p in picked:
            if not (c["end"] + min_gap <= p["start"] or
                    p["end"] + min_gap <= c["start"]):
                ok = False
                break
        if ok:
            picked.append(c)
        if len(picked) >= jumlah:
            break
    return picked


def write_srt(words, offset, path):
    # Kelompokkan kata jadi cue maksimal ±7 kata / 3.5 detik
    cues, cur, cur_start = [], [], None
    for w in words:
        if cur_start is None:
            cur_start = w["s"]
        cur.append(w)
        if len(cur) >= 7 or (w["e"] - cur_start) >= 3.5:
            cues.append(cur)
            cur, cur_start = [], None
    if cur:
        cues.append(cur)
    with open(path, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            s = cue[0]["s"] - offset
            e = cue[-1]["e"] - offset
            f.write(f"{i}\n{fmt_srt(s)} --> {fmt_srt(e)}\n"
                    f"{' '.join(w['w'] for w in cue)}\n\n")


def cut_clip(video, cand, out_path, fmt, burn_srt=None):
    start, dur = cand["start"], cand["duration"]
    base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}",
            "-i", video, "-t", f"{dur:.2f}"]
    if fmt == "vertikal":
        w, h = probe_size(video)
        if abs(w / h - 9 / 16) < 0.05:
            vf = "scale=1080:1920:force_original_aspect_ratio=disable"
        else:
            vf = ("split[a][b];"
                  "[a]scale=1080:1920:force_original_aspect_ratio=increase,"
                  "crop=1080:1920,boxblur=24:6[bg];"
                  "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
        if burn_srt:
            vf += f",subtitles='{burn_srt}'"
        cmd = base + ["-vf", vf, "-c:v", "libx264", "-crf", "20",
                      "-preset", "medium", "-c:a", "aac", "-b:a", "160k", out_path]
    else:
        if burn_srt:
            cmd = base + ["-vf", f"subtitles='{burn_srt}'", "-c:v", "libx264",
                          "-crf", "20", "-preset", "medium",
                          "-c:a", "aac", "-b:a", "160k", out_path]
        else:
            cmd = base + ["-c:v", "libx264", "-crf", "20", "-preset", "medium",
                          "-c:a", "aac", "-b:a", "160k", out_path]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg potong gagal: " + p.stderr[-500:])


def main():
    ap = argparse.ArgumentParser(description="Potong video panjang jadi klip Shorts + peta timestamp penting.")
    ap.add_argument("video")
    ap.add_argument("--jumlah", type=int, default=5, help="jumlah klip (default 5)")
    ap.add_argument("--min-dur", type=float, default=20, help="durasi minimum klip detik (default 20)")
    ap.add_argument("--max-dur", type=float, default=60, help="durasi maksimum klip detik (default 60)")
    ap.add_argument("--model", default="small", help="model whisper: tiny/base/small/medium/large-v3-turbo (default small)")
    ap.add_argument("--lang", default="id", help="bahasa: id/en/auto (default id)")
    ap.add_argument("--format", choices=["vertikal", "asli"], default="vertikal",
                    help="vertikal 9:16 latar blur untuk Shorts, atau asli (default vertikal)")
    ap.add_argument("--caption", action="store_true", help="bakar caption SRT ke dalam video klip")
    ap.add_argument("--out", default=None, help="folder keluaran (default: hasil-<nama video>)")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        sys.exit(f"File tidak ditemukan: {video}")
    stem = os.path.splitext(os.path.basename(video))[0]
    outdir = args.out or os.path.join(os.path.dirname(video), f"hasil-{stem}")
    os.makedirs(outdir, exist_ok=True)

    total = probe_duration(video)
    print(f"Video: {os.path.basename(video)} ({fmt_ts(total)})")
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        print("1/4 Ekstrak audio...")
        extract_audio(video, wav)
        print(f"2/4 Transkrip (model {args.model}, bahasa {args.lang})...")
        segs, detected = transcribe(wav, args.model, args.lang)
        print(f"    {len(segs)} segmen bicara, bahasa terdeteksi: {detected}")
        if not segs:
            sys.exit("Tidak ada bicara yang terdeteksi; batal.")
        print("3/4 Menilai momen penting...")
        cands = build_candidates(segs, args.min_dur, args.max_dur)
        picked = pick_top(cands, args.jumlah)
        if not picked:
            sys.exit("Tidak ada jendela yang memenuhi rentang durasi.")
        print(f"    {len(cands)} kandidat dinilai, {len(picked)} klip dipilih.")
        print("4/4 Memotong klip...")
        results = []
        for rank, c in enumerate(sorted(picked, key=lambda x: -x["score"]), 1):
            name = f"klip-{rank:02d}"
            srt_rel = f"{name}.srt"
            srt_path = os.path.join(outdir, srt_rel)
            write_srt(c["_words"], c["start"], srt_path)
            out_path = os.path.join(outdir, f"{name}.mp4")
            burn = srt_path.replace("'", "") if args.caption else None
            cut_clip(video, c, out_path, args.format, burn_srt=burn)
            results.append({"rank": rank, "file": f"{name}.mp4", "srt": srt_rel,
                            "start": c["start"], "end": c["end"],
                            "start_ts": fmt_ts(c["start"]), "end_ts": fmt_ts(c["end"]),
                            "duration": c["duration"], "score": c["score"],
                            "text": c["text"]})
            print(f"    {name}.mp4  {fmt_ts(c['start'])}-{fmt_ts(c['end'])}  "
                  f"({c['duration']:.0f}s, skor {c['score']})")

    # Laporan: klip terpilih + peta semua momen penting (15 besar, urut waktu)
    top_all = sorted(pick_top(cands, 15), key=lambda x: x["start"])
    picked_keys = {(r["start"], r["end"]) for r in results}
    with open(os.path.join(outdir, "timestamps.json"), "w", encoding="utf-8") as f:
        json.dump({"video": os.path.basename(video), "durasi_total": round(total, 2),
                   "klip": results,
                   "momen_penting": [
                       {"start_ts": fmt_ts(c["start"]), "end_ts": fmt_ts(c["end"]),
                        "start": c["start"], "end": c["end"], "score": c["score"],
                        "terpilih": (c["start"], c["end"]) in picked_keys,
                        "text": c["text"]} for c in top_all]},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "timestamps.md"), "w", encoding="utf-8") as f:
        f.write(f"# Peta Timestamp Penting — {os.path.basename(video)}\n\n")
        f.write(f"Durasi total {fmt_ts(total)}. Bahasa terdeteksi: {detected}. "
                f"Model transkrip: {args.model}.\n\n")
        f.write("## Klip yang dibuat\n\n")
        for r in results:
            f.write(f"### {r['file']} — {r['start_ts']} s.d. {r['end_ts']} "
                    f"({r['duration']:.0f} detik, skor {r['score']})\n\n"
                    f"> {r['text']}\n\n")
        f.write("## Semua momen penting (15 besar, urut waktu)\n\n")
        for c in top_all:
            mark = "TERPILIH" if (c["start"], c["end"]) in picked_keys else "-"
            f.write(f"- **{fmt_ts(c['start'])} - {fmt_ts(c['end'])}** "
                    f"(skor {c['score']}, {mark}): {c['text'][:140]}...\n")
    print(f"\nSelesai. Hasil di: {outdir}")
    print("Laporan: timestamps.md + timestamps.json")


if __name__ == "__main__":
    main()
