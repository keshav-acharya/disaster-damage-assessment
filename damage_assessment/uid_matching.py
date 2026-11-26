
import json
import csv
from pathlib import Path

LABELS_DIR = Path(
    r"D:\Disaster-Damage-Assessment\data\raw\train\labels"
)


def load_buildings(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    features = data.get("features", {}).get("xy", [])

    buildings = {}

    for feature in features:
        properties = feature.get("properties", {})
        uid = properties.get("uid")

        if uid is not None:
            buildings[str(uid)] = properties.get("subtype", "")

    return buildings


def match_buildings():
    post_files = sorted(LABELS_DIR.glob("*_post_disaster.json"))

    if not post_files:
        print("No post-disaster JSON files found!")
        return

    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    pair_results = []
    building_results = []

    total_pre = 0
    total_post = 0
    total_matched = 0
    missing_pre_files = 0

    for post_path in post_files:
        pre_path = post_path.with_name(
            post_path.name.replace(
                "_post_disaster.json",
                "_pre_disaster.json"
            )
        )

        if not pre_path.exists():
            missing_pre_files += 1
            continue

        try:
            pre_buildings = load_buildings(pre_path)
            post_buildings = load_buildings(post_path)

        except (json.JSONDecodeError, OSError) as error:
            print(f"Error reading {post_path.name}: {error}")
            continue

        pre_uids = set(pre_buildings)
        post_uids = set(post_buildings)

        matched_uids = pre_uids.intersection(post_uids)

        unmatched_pre = pre_uids - post_uids
        unmatched_post = post_uids - pre_uids

        total_pre += len(pre_uids)
        total_post += len(post_uids)
        total_matched += len(matched_uids)

        pair_results.append([
            post_path.stem.replace("_post_disaster", ""),
            len(pre_uids),
            len(post_uids),
            len(matched_uids),
            len(unmatched_pre),
            len(unmatched_post)
        ])

        for uid in sorted(matched_uids):
            building_results.append([
                post_path.stem.replace("_post_disaster", ""),
                uid,
                pre_buildings[uid],
                post_buildings[uid]
            ])

    # Save pair-level validation
    pair_csv = output_dir / "uid_pair_validation.csv"

    with open(pair_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image Pair",
            "Pre Buildings",
            "Post Buildings",
            "Matched UIDs",
            "Unmatched Pre",
            "Unmatched Post"
        ])
        writer.writerows(pair_results)

    # Save individual building matches
    matches_csv = output_dir / "building_uid_matches.csv"

    with open(matches_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image Pair",
            "Building UID",
            "Pre Damage Label",
            "Post Damage Label"
        ])
        writer.writerows(building_results)

    print("\n========== UID MATCHING RESULTS ==========")
    print(f"Post-disaster files found: {len(post_files)}")
    print(f"Valid image pairs: {len(pair_results)}")
    print(f"Missing pre-disaster files: {missing_pre_files}")
    print(f"Pre-disaster buildings: {total_pre}")
    print(f"Post-disaster buildings: {total_post}")
    print(f"Matched building UIDs: {total_matched}")

    print(f"\nPair validation saved: {pair_csv}")
    print(f"Building matches saved: {matches_csv}")


if __name__ == "__main__":
    match_buildings()
