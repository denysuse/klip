# klip — Long Video to Shorts Clipper with an Important-Timestamp Map

Turn one long video (podcast, tutorial, livestream, talk) into several
ready-to-post vertical Shorts, plus a readable map of the most important
timestamps in the whole video. Transcription and scoring run **fully
locally** — no uploads, no API keys.

Satu perintah: video panjang masuk, klip Shorts vertikal + peta timestamp
penting keluar. Semua proses lokal memakai faster-whisper dan ffmpeg.

## What it does

1. Extracts audio and transcribes it locally with
   [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (word-level timestamps).
2. Builds candidate windows from speech-segment boundaries and scores each
   one heuristically: speech density, hook words (Indonesian + English),
   numbers, questions, a natural pause before the start, an ending on a
   sentence boundary, and transcript confidence.
3. Cuts the best non-overlapping windows with ffmpeg into 1080×1920
   vertical clips (blurred-background fill), writes a per-clip SRT, and
   produces a timestamp report of the top moments — including good moments
   that were *not* cut, so a human can still pick them in seconds.

The score is a selection aid, not a verdict. The timestamp map is the real
product: it lets you skim a 2-hour video's best moments in one page.

## Quick start

Requirements: Python 3.10+, `ffmpeg` and `ffprobe` on PATH.

```bash
git clone https://github.com/denysuse/klip.git
cd klip
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python klip.py my-long-video.mp4 --jumlah 5
```

The Whisper model downloads once on first use (base ≈ 150 MB,
small ≈ 500 MB), then everything runs offline.

## Output

Written to `hasil-<video name>/` next to the input (or `--out DIR`):

- `klip-01.mp4`, `klip-02.mp4`, … — best clips first, 1080×1920 vertical
  (`--format asli` keeps the source aspect ratio)
- `klip-01.srt`, … — per-clip captions, ready to edit or upload
- `timestamps.md` — human-readable report: the clips made + top 15
  important moments in time order, each with its spoken text
- `timestamps.json` — the same data for machines

## Options

| Option | Default | Notes |
|---|---|---|
| `--jumlah` | 5 | how many clips to cut |
| `--min-dur` / `--max-dur` | 20 / 60 | clip duration range, seconds |
| `--model` | small | Whisper model: tiny, base, small, medium, large-v3-turbo. Bigger = more accurate, slower. Use small or better for Indonesian |
| `--lang` | id | speech language: `id`, `en`, or `auto` |
| `--format` | vertikal | `vertikal` = 9:16 blurred fill for Shorts/Reels/TikTok; `asli` = source aspect |
| `--caption` | off | burn the SRT into the clip picture |

## Tested

On a 24-minute podcast video (base model, CPU): 201 speech segments,
1,057 candidate windows scored, 3 vertical clips cut, plus the full
timestamp map — 8 minutes 37 seconds end to end.

Known limitation: a high-scoring window can still catch a "like and
subscribe" call-to-action segment (spoken numbers inflate its score).
That is exactly why the timestamp map exists — final picks stay human.

## Pairs well with

- [video-short](https://github.com/petergyang/video-short) — an agent
  skill that polishes a raw talking-head recording into a finished Short
  (B-roll, motion graphics, frame-by-frame review page). `klip` mines the
  moments; `video-short` polishes them.

## License

MIT — see [LICENSE](LICENSE).
