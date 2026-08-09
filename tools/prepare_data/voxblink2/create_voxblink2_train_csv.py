"""
Create the sslsv train .csv for VoxBlink2.

On Jean Zay the corpus is already built and available (read-only) in
$DSDIR/VoxBlink2/audio, laid out as <speaker>/<video>/<utterance>.wav.

Walking ~10M files on Lustre is metadata-bound, so speakers are scanned in
parallel and rows are streamed to the .csv instead of being kept in memory.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--audio_dir",
        default="/lustre/fsmisc/dataset/VoxBlink2/audio",
        type=Path,
        help="Root of the corpus (<speaker>/<video>/<utterance>.wav)",
    )
    parser.add_argument(
        "--prefix",
        default="voxblink2",
        type=str,
        help="Prepended to every path, relative to dataset.base_path",
    )
    parser.add_argument("--output", default="voxblink2_train.csv", type=Path)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--min_utterances",
        default=0,
        type=int,
        help="Discard speakers with fewer utterances",
    )
    parser.add_argument(
        "--max_speakers",
        default=None,
        type=int,
        help="Keep only the first N speakers (to build a subset)",
    )
    return parser.parse_args()


def scan_speaker(speaker: str) -> Tuple[str, List[str]]:
    files = []
    speaker_dir = os.path.join(AUDIO_DIR, speaker)
    for video in os.scandir(speaker_dir):
        if not video.is_dir():
            continue
        for utterance in os.scandir(video.path):
            if utterance.name.endswith(".wav"):
                files.append(f"{PREFIX}/{speaker}/{video.name}/{utterance.name}")
    return speaker, sorted(files)


def main():
    global AUDIO_DIR, PREFIX

    args = parse_args()
    AUDIO_DIR = str(args.audio_dir)
    PREFIX = args.prefix.rstrip("/")

    speakers = sorted(e.name for e in os.scandir(args.audio_dir) if e.is_dir())
    if args.max_speakers is not None:
        speakers = speakers[: args.max_speakers]
    print(f"Scanning {len(speakers)} speakers in {args.audio_dir}", flush=True)

    nb_utterances, nb_speakers, nb_discarded = 0, 0, 0

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Speaker"])

        with mp.Pool(processes=args.num_workers) as pool:
            for i, (speaker, files) in enumerate(
                pool.imap(scan_speaker, speakers, chunksize=32)
            ):
                if len(files) < args.min_utterances:
                    nb_discarded += 1
                    continue
                writer.writerows((file, speaker) for file in files)
                nb_utterances += len(files)
                nb_speakers += 1

                if (i + 1) % 5000 == 0:
                    print(
                        f"{i + 1}/{len(speakers)} speakers"
                        f" - {nb_utterances} utterances",
                        flush=True,
                    )

    print(
        f"{nb_utterances} utterances - {nb_speakers} speakers"
        f" ({nb_discarded} discarded) -> {args.output}"
    )


if __name__ == "__main__":
    main()
