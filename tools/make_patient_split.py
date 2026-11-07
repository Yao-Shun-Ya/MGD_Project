"""Build a patient-independent split from MGD-1K metadata.

The public MGD-1K filenames contain a numeric patient/study id. We group by
that id, stratify patients by their maximum expert Meiboscore, and write the
engine-compatible dataset_splits.json plus a metadata CSV.

Outputs:
  data/dataset_splits.json        (patient-independent holdout)
  data/dataset_metadata.csv
  data/official_test_stems.txt
  data/split_report.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
META = ROOT / "MGD1k-main" / "分数参考"
SPLITS = DATA / "dataset_splits.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DUPLICATE_STEM = "DMI_OS_LOWER_REFLECTED_IR_14933989_001"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stems_signature(stems: list[str]) -> str:
    payload = "\n".join(sorted(str(item) for item in stems))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assignment_signature(train: list[str], val: list[str], test: list[str]) -> str:
    payload = (
        "train:" + ",".join(sorted(train))
        + "|val:" + ",".join(sorted(val))
        + "|test:" + ",".join(sorted(test))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def patient_id(stem: str) -> str:
    match = re.search(r"(\d{6,})", stem)
    return match.group(1) if match else stem


def parse_eye_eyelid(stem: str) -> tuple[str, str]:
    eye = "OD" if "_OD_" in stem else "OS" if "_OS_" in stem else "unknown"
    eyelid = "upper" if "UPPER" in stem.upper() else "lower" if "LOWER" in stem.upper() else "unknown"
    return eye, eyelid


def main() -> int:
    scores: dict[str, int] = {}
    with open(META / "Images with Meiboscore.csv", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            name = row[0].strip()
            try:
                value = int(float(row[1]))
            except ValueError:
                continue
            stem = Path(name).stem
            scores[stem] = value

    files = [
        p for p in (DATA / "images").iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem != DUPLICATE_STEM
    ]
    stems = sorted(p.stem for p in files)
    missing = [stem for stem in stems if stem not in scores]
    extra = [stem for stem in scores if stem not in set(stems) and stem != DUPLICATE_STEM]
    print(f"images={len(stems)} scored={len(scores)} missing_scores={len(missing)} extra_scores={len(extra)}")
    if missing:
        print("missing score examples:", missing[:5])

    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        groups[patient_id(stem)].append(stem)
    group_score = {pid: max(scores.get(s, 1) for s in members) for pid, members in groups.items()}
    print(f"patients={len(groups)} images={sum(len(v) for v in groups.values())}")

    rng = random.Random(42)
    # Stratify by maximum expert Meiboscore; assign whole patients to test/val/train.
    total = len(stems)
    target_test = int(round(total * 0.15))
    target_val = int(round(total * 0.15))
    by_score: dict[int, list[str]] = defaultdict(list)
    for pid, score in group_score.items():
        by_score[score].append(pid)

    test: list[str] = []
    val: list[str] = []
    train: list[str] = []
    for score in sorted(by_score):
        pids = by_score[score]
        rng.shuffle(pids)
        stratum_images = sum(len(groups[pid]) for pid in pids)
        test_target = int(round(stratum_images * target_test / total))
        val_target = int(round(stratum_images * target_val / total))
        test_count = val_count = 0
        for pid in pids:
            members = groups[pid]
            if test_count < test_target:
                test.extend(members)
                test_count += len(members)
            elif val_count < val_target:
                val.extend(members)
                val_count += len(members)
            else:
                train.extend(members)

    # Move surplus from test/val to train (greedy, whole patients) to hit ~70/15/15.
    def move_surplus(source: list[str], target_count: int, dest: list[str]) -> None:
        source.sort()
        while len(source) > target_count + 8:
            stem = source.pop()
            dest.append(stem)

    move_surplus(test, target_test, train)
    move_surplus(val, target_val, train)

    train, val, test = sorted(train), sorted(val), sorted(test)
    assert not (set(train) & set(test)) and not (set(train) & set(val)) and not (set(val) & set(test))
    assert sum(len(x) for x in (train, val, test)) == total

    # Patient overlap audit
    pid_train = {patient_id(s) for s in train}
    pid_val = {patient_id(s) for s in val}
    pid_test = {patient_id(s) for s in test}
    overlap = (pid_train & pid_test) | (pid_train & pid_val) | (pid_val & pid_test)
    if overlap:
        raise RuntimeError(f"patient overlap remains: {overlap}")

    # Official 200-image Meiboscore test list
    official = [line.strip() for line in (META / "200 test Images used for Meiboscore Classification.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    official_stems = [Path(name).stem for name in official if Path(name).stem in set(stems)]
    official_pids = {patient_id(s) for s in official_stems}

    signature = stems_signature(stems)
    assignment = assignment_signature(train, val, test)
    score_counts = {split: dict(sorted(Counter(scores.get(s, -1) for s in members).items())) for split, members in (("train", train), ("val", val), ("test", test))}
    patient_counts = {"train": len(pid_train), "val": len(pid_val), "test": len(pid_test)}

    payload = {
        "format_version": 1,
        "signature": signature,
        "dataset_size": total,
        "created_at": utc_now(),
        "entries": {
            "holdout": {
                "spec": {"mode": "holdout", "val_ratio": 0.15, "test_ratio": 0.15, "seed": 42, "n_folds": 5, "fold": 0},
                "signature": signature,
                "train": train,
                "val": val,
                "test": test,
                "warnings": [],
                "created_at": utc_now(),
                "split_protocol": "patient-independent (grouped by numeric study id, stratified by max Meiboscore)",
                "assignment_signature": assignment,
                "patient_counts": patient_counts,
                "meiboscore_distribution": score_counts,
            }
        },
    }
    SPLITS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(DATA / "dataset_metadata.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "patient_id", "eye", "eyelid", "meiboscore", "split"])
        split_of = {**{s: "train" for s in train}, **{s: "val" for s in val}, **{s: "test" for s in test}}
        for stem in stems:
            eye, eyelid = parse_eye_eyelid(stem)
            writer.writerow([stem, patient_id(stem), eye, eyelid, scores.get(stem, ""), split_of[stem]])

    (DATA / "official_test_stems.txt").write_text("\n".join(official_stems) + "\n", encoding="utf-8")

    report = {
        "signature": signature,
        "assignment_signature": assignment,
        "dataset_size": total,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "patient_counts": patient_counts,
        "meiboscore_distribution": score_counts,
        "official_test": {
            "images": len(official_stems),
            "patients": len(official_pids),
            "images_in_data": len(official_stems),
            "patient_overlap_with_train": len(official_pids & pid_train),
            "patient_overlap_with_val": len(official_pids & pid_val),
            "patient_overlap_with_test": len(official_pids & pid_test),
            "note": "The official 200-image list is NOT patient-independent from the remaining images; 326/800 non-test images come from official-test patients.",
        },
    }
    (DATA / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
