import os
import random
import argparse
from pathlib import Path


VOX_ROOT = Path("voxceleb2")
OUTPUT_FILE = "voxceleb2_test_fusion2"
N_SPEAKERS = 50
N_NONTARGET = 20000
N_TARGET = 20000
SEED = 17


def create_voxceleb2_trials():
    random.seed(SEED)

    # Step 1: Get speakers and their utterances
    speakers = sorted([d.name for d in VOX_ROOT.iterdir() if d.is_dir()])
    selected_speakers = random.sample(speakers, k=N_SPEAKERS)

    speaker_to_utts = {}

    for spk in selected_speakers:
        spk_path = VOX_ROOT / spk
        utts = list(spk_path.glob("*/*.wav"))
        if len(utts) >= 2:
            speaker_to_utts[spk] = utts

    already_generated = set()

    # Step 2: Generate target trials
    target_trials = []
    while len(target_trials) < N_TARGET:
        spk = random.choice(list(speaker_to_utts.keys()))
        u1 = random.choice(speaker_to_utts[spk])
        u2 = random.choice(speaker_to_utts[spk])
        if f"{u1}_{u2}" in already_generated:
            continue
        target_trials.append((1, u1, u2))
        already_generated.add(f"{u1}_{u2}")

    # Step 3: Generate non-target trials
    nontarget_trials = []
    while len(nontarget_trials) < N_NONTARGET:
        spk1, spk2 = random.sample(list(speaker_to_utts.keys()), 2)
        u1 = random.choice(speaker_to_utts[spk1])
        u2 = random.choice(speaker_to_utts[spk2])
        if f"{u1}_{u2}" in already_generated:
            continue
        nontarget_trials.append((0, u1, u2))
        already_generated.add(f"{u1}_{u2}")

    # Step 4: Combine and save
    all_trials = target_trials + nontarget_trials
    random.shuffle(all_trials)

    with open(OUTPUT_FILE, "w") as f:
        for label, u1, u2 in all_trials:
            u1_str = str(u1.relative_to(VOX_ROOT))
            u2_str = str(u2.relative_to(VOX_ROOT))
            f.write(f"{label} voxceleb2/{u1_str} voxceleb2/{u2_str}\n")

    print(f"Trial file written to {OUTPUT_FILE} with {len(all_trials)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Path to store datasets.")
    args = parser.parse_args()

    os.chdir(args.output_path)

    create_voxceleb2_trials()
