import argparse
import csv
from pathlib import Path

def convert_file(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r") as infile, open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["File", "Speaker"])

        for line in infile:
            parts = line.strip().split()
            rel_path, speaker_id = parts[0], parts[1]
            full_path = f"voxceleb2/{rel_path}"
            writer.writerow([full_path, speaker_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input text file")
    parser.add_argument("output_path", help="Path to output CSV file")
    args = parser.parse_args()

    convert_file(args.input_path, args.output_path)
