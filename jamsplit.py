#!/usr/bin/env python3
"""Split long bluegrass jam recordings into individual, titled tracks.

Pipeline:
  1. Tag every 10 s window with an AudioSet classifier (speech vs. bluegrass).
  2. Transcribe the recording with Whisper (talk between songs, sung lyrics).
  3. Ask Claude to turn the timeline + transcript into song boundaries and titles.
  4. Cut the songs out of the original file with ffmpeg.

Usage:
  python jamsplit.py local RECORDING.mp3 [--outdir DIR]
  python jamsplit.py drive --inbox "Jam Inbox" --output "Jam Tracks"
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
WIN = 10  # seconds per classifier window
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
MIN_SONG = 60  # seconds
PAD = float(os.environ.get("PAD_SECONDS", 10))  # extra audio kept before/after each song
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".amr", ".3gp"}


def log(*a):
    print(*a, flush=True)


def fmt(t):
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def decode(path, wav):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                    "-ac", "1", "-ar", str(SR), str(wav)], check=True)
    y, _ = sf.read(wav, dtype="float32")
    return y


def tag(y):
    """Per-window probabilities of speech and bluegrass/country music."""
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    torch.set_num_threads(os.cpu_count() or 2)
    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL).eval()
    ids = {l: i for i, l in model.config.id2label.items()}
    n = int(np.ceil(len(y) / (WIN * SR)))
    probs = []
    for b in range(0, n, 8):
        chunks = [y[i * WIN * SR:(i + 1) * WIN * SR] for i in range(b, min(n, b + 8))]
        x = fe(chunks, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            probs.append(torch.sigmoid(model(**x).logits).numpy())
        if b % 80 == 0:
            log(f"  tagging {fmt(b * WIN)} / {fmt(n * WIN)}")
    p = np.concatenate(probs)
    speech = p[:, ids["Speech"]]
    music = p[:, ids["Bluegrass"]] + 0.5 * p[:, ids["Country"]]
    singing = p[:, ids["Singing"]]
    return speech, music, singing


def candidate_segments(speech, music):
    """Rough song boundaries: runs where bluegrass outweighs speech."""
    from scipy.ndimage import median_filter

    song = median_filter((music - speech > 0).astype(int), size=3) > 0
    segs = []
    for i, s in enumerate(song):
        if not s:
            continue
        if segs and i - segs[-1][1] <= 1:
            segs[-1][1] = i + 1
        else:
            segs.append([i, i + 1])
    return [(a * WIN, b * WIN) for a, b in segs if (b - a) * WIN >= MIN_SONG]


def transcribe(y, speech, singing, cands):
    """Transcribe windows that likely contain talk or singing, plus the lead-in to each song.

    Whisper's own voice detection treats a loud jam as one long block of speech, and it
    hallucinates (slowly) on instrumentals, so we pick the windows ourselves.
    """
    from faster_whisper import WhisperModel

    keep = (speech >= 0.2) | (singing >= 0.3)
    for a, _ in cands:
        i = a // WIN
        keep[max(0, i - 3):i + 1] = True
    chunks, i = [], 0
    while i < len(keep):
        if keep[i]:
            j = i
            while j < len(keep) and keep[j] and j - i < 3:
                j += 1
            chunks.append((i * WIN, j * WIN))
            i = j
        else:
            i += 1
    log(f"  transcribing {len(chunks)} chunks ({fmt(sum(b - a for a, b in chunks))} of audio)")
    m = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 2)
    out = []
    for k, (a, b) in enumerate(chunks):
        segs, _ = m.transcribe(y[a * SR:b * SR], language="en", vad_filter=False, beam_size=1,
                               temperature=0.0, condition_on_previous_text=False,
                               without_timestamps=True, max_new_tokens=120)
        text = " ".join(s.text.strip() for s in segs
                        if s.no_speech_prob < 0.6 and s.compression_ratio < 2.4)
        if text:
            out.append((a, b, text))
        if k % 50 == 49:
            log(f"  transcribed {k + 1}/{len(chunks)}")
    return out


PROMPT = """You are splitting a long recording of an amateur bluegrass jam session into individual songs/tunes.

Recording length: {length}.

## Classifier timeline
One line per 10 s window: start time, then S = speech probability, B = bluegrass-music probability, V = singing probability (0-9 scale).
Talk between songs usually shows high S with low B. People often noodle on instruments while chatting, so B alone is not decisive.

{timeline}

## Automatic first guess at songs (may wrongly merge two songs or split one song)
{candidates}

## Whisper transcript (noisy; lyrics of sung songs and chat between songs both appear here; Whisper sometimes hallucinates text during instrumentals)
{transcript}

## Task
Work out where each song or tune starts and ends, and what it is. Use the chat (people often name the tune, the key, or who takes the next one: "let's do Salt Creek in A"), the lyrics, and the classifier timeline.
- start = when the music of the song begins (include a count-in or kick-off); end = when the song finishes (before the chat starts).
- Exclude talk, tuning, and noodling between songs. Skip anything under 60 s that is not clearly a song.
- Title: use the standard title of the bluegrass/old-time/folk/country song or fiddle tune if you can identify it from lyrics or chat. If you cannot identify it with reasonable confidence, use null.
- A song named in the chat shortly before a song starts is the strongest evidence, stronger than a guess from garbled lyrics. Whisper often mishears names, so map them to the real song they most likely are (e.g. "Street Line Cannonball" -> "Streamline Cannonball", "Marulita by Warren Zeeblon" -> "Carmelita").
- Times in seconds from the start of the recording.

Reply with only a JSON array, no prose:
[{{"start": 410, "end": 530, "title": "Salt Creek", "confidence": "high|medium|low", "evidence": "short reason"}}]
"""


def plan_with_claude(length, speech, music, singing, cands, transcript):
    import anthropic

    timeline = "\n".join(
        f"{fmt(i * WIN)} S{min(9, int(s * 10))} B{min(9, int(m * 10))} V{min(9, int(v * 10))}"
        for i, (s, m, v) in enumerate(zip(speech, music, singing)))
    cand_txt = "\n".join(f"{fmt(a)} - {fmt(b)}" for a, b in cands) or "(none)"
    tr = "\n".join(f"[{fmt(a)}-{fmt(b)}] {t}" for a, b, t in transcript) or "(no speech detected)"
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": PROMPT.format(
            length=fmt(length), timeline=timeline, candidates=cand_txt, transcript=tr)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    songs = json.loads(m.group(0))
    songs = [s for s in songs if float(s["end"]) - float(s["start"]) >= MIN_SONG]
    return sorted(songs, key=lambda s: float(s["start"]))


def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|]+', "", s).strip()
    return s[:80] or "Untitled"


def cut(src, songs, outdir, album, length):
    outdir.mkdir(parents=True, exist_ok=True)
    copy = src.suffix.lower() == ".mp3"
    lines = []
    for n, s in enumerate(songs, 1):
        title = s.get("title") or f"Unknown tune {n}"
        name = f"{n:02d} - {safe_name(title)}.mp3"
        start, end = max(0.0, float(s["start"]) - PAD), min(length, float(s["end"]) + PAD)
        codec = ["-c:a", "copy"] if copy else ["-c:a", "libmp3lame", "-q:a", "2"]
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", str(start), "-to", str(end), "-i", str(src), "-map", "0:a",
                        *codec, "-id3v2_version", "3",
                        "-metadata", f"title={title}", "-metadata", f"album={album}",
                        "-metadata", f"track={n}/{len(songs)}",
                        str(outdir / name)], check=True)
        lines.append(f"{name}\t{fmt(start)}-{fmt(end)}\t{s.get('confidence', '')}\t{s.get('evidence', '')}")
        log(f"  {name}  [{fmt(start)}-{fmt(end)}]")
    (outdir / "tracklist.txt").write_text(
        f"{album}\n\nfile\tposition in original\tconfidence\tevidence\n" + "\n".join(lines) + "\n")


def process(src, outdir, workdir):
    log(f"== {src.name}")
    y = decode(src, workdir / "audio16k.wav")
    length = len(y) / SR
    log(f"  length {fmt(length)}; tagging")
    speech, music, singing = tag(y)
    cands = candidate_segments(speech, music)
    log(f"  {len(cands)} candidate songs")
    transcript = transcribe(y, speech, singing, cands)
    del y
    log("  asking Claude for boundaries and titles")
    try:
        songs = plan_with_claude(length, speech, music, singing, cands, transcript)
    except Exception as e:  # fall back to the classifier's guess
        log(f"  Claude step failed ({e}); using classifier segments without titles")
        songs = [{"start": a, "end": b, "title": None, "confidence": "low",
                  "evidence": "classifier only"} for a, b in cands]
    cut(src, songs, outdir, album=src.stem, length=length)
    return songs


def rclone(*args, capture=False):
    r = subprocess.run(["rclone", *args], check=True, text=True,
                       stdout=subprocess.PIPE if capture else None)
    return r.stdout if capture else None


def run_drive(inbox, output):
    files = [f for f in rclone("lsf", f"gdrive:{inbox}", "--files-only", "--max-depth", "1",
                               capture=True).splitlines()
             if Path(f).suffix.lower() in AUDIO_EXTS]
    if not files:
        log("nothing to do")
        return
    done = {d.rstrip("/") for d in rclone("lsf", f"gdrive:{output}", "--dirs-only",
                                           capture=True).splitlines()}
    failed = []
    for f in files:
        if Path(f).stem in done:  # e.g. re-uploaded by a phone sync app
            log(f"== {f} already processed; moving to processed/")
            rclone("moveto", f"gdrive:{inbox}/{f}", f"gdrive:{inbox}/processed/{f}")
            continue
        work = Path(tempfile.mkdtemp())
        try:
            src = work / f
            rclone("copyto", f"gdrive:{inbox}/{f}", str(src))
            out = work / "out"
            process(src, out, work)
            rclone("copy", str(out), f"gdrive:{output}/{Path(f).stem}")
            rclone("moveto", f"gdrive:{inbox}/{f}", f"gdrive:{inbox}/processed/{f}")
        except Exception as e:
            log(f"!! failed on {f}: {e}")
            failed.append(f)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if failed:
        sys.exit(f"failed: {failed}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("local")
    lp.add_argument("file", type=Path)
    lp.add_argument("--outdir", type=Path)
    dp = sub.add_parser("drive")
    dp.add_argument("--inbox", default="Jam Inbox")
    dp.add_argument("--output", default="Jam Tracks")
    a = ap.parse_args()
    if a.cmd == "local":
        out = a.outdir or a.file.with_suffix("")
        with tempfile.TemporaryDirectory() as w:
            process(a.file, out, Path(w))
    else:
        run_drive(a.inbox, a.output)


if __name__ == "__main__":
    main()
