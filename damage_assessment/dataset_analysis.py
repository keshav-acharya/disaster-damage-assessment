
import json
from pathlib import Path
from collections import Counter
import csv

# Path to the existing xBD dataset on your D drive
DATASET_PATH = Path(
    r"D:\Disaster-Damage-Assessment\data\raw\train\labels"
)

# Damage categories
DAMAGE_CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
    "un-classified"
]


def analyze_dataset():
    if not DATASET_PATH.exists():
        print("ERROR: Dataset labels folder not found!")
        return

    json_files = list(DATASET_PATH.glob("*_post_disaster.json"))

    if not json_files:
        print("No post-disaster JSON files found!")
        return

    counts = Counter()

    print(f"Found {len(json_files)} post-disaster annotation files.")
    print("Analyzing damage categories...\n")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            features = data.get("features", {}).get("xy", [])

            for feature in features:
                subtype = feature.get("properties", {}).get("subtype")

                if subtype in DAMAGE_CLASSES:
                    counts[subtype] += 1

        except (json.JSONDecodeError, OSError) as error:
            print(f"Could not read {file_path.name}: {error}")

    total = sum(counts.values())

    print("\n========== DAMAGE DISTRIBUTION ==========")
    print(f"Total building annotations: {total}\n")

    for damage_class in DAMAGE_CLASSES:
        count = counts[damage_class]
        percentage = (count / total * 100) if total else 0

        print(
            f"{damage_class:15} : "
            f"{count:8} buildings "
            f"({percentage:.2f}%)"
        )

    # Save results in the repository
    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "damage_distribution.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Damage Class", "Building Count", "Percentage"])

        for damage_class in DAMAGE_CLASSES:
            count = counts[damage_class]
            percentage = (count / total * 100) if total else 0
            writer.writerow([damage_class, count, round(percentage, 2)])

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    analyze_dataset()
