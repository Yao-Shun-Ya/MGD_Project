"""Create split files for the leakage study.

- patient_holdout.json: copy of the current patient-independent split
- official_leaky.json: test = official 200; train/val = remaining 799 split
  randomly at image level (the common, leaky practice)
- clean_official_stems.txt: official test images whose patient is NOT in the
  patient-independent training set (a clean external subset)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "splits"
OUT.mkdir(parents=True, exist_ok=True)
META = ROOT / "MGD1k-main" / "分数参考"


def sig(stems: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(stems)).encode("utf-8")).hexdigest()


def assignment_sig(train: list[str], val: list[str], test: list[str]) -> str:
    payload = "train:" + ",".join(sorted(train)) + "|val:" + ",".join(sorted(val)) + "|test:" + ",".join(sorted(test))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pid(stem: str) -> str:
    m = re.search(r"(\d{6,})", stem)
    return m.group(1) if m else stem


def main() -> int:
    all_stems = sorted(p.stem for p in (DATA / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    official = [Path(line.strip()).stem for line in (META / "200 test Images used for Meiboscore Classification.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    official = [s for s in official if s in set(all_stems)]
    rest = [s for s in all_stems if s not in set(official)]
    rng = random.Random(42)
    rng.shuffle(rest)
    n_val = int(round(len(all_stems) * 0.15))
    val, train = rest[:n_val], rest[n_val:]
    assert not (set(train) & set(test := official)) and not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(all_stems)

    payload = {
        "format_version": 1,
        "signature": sig(all_stems),
        "dataset_size": len(all_stems),
        "created_at": "2026-11-07T00:00:00+00:00",
        "entries": {
            "holdout": {
                "spec": {"mode": "holdout", "val_ratio": 0.15, "test_ratio": 0.15, "seed": 42, "n_folds": 5, "fold": 0},
                "signature": sig(all_stems),
                "train": sorted(train),
                "val": sorted(val),
                "test": sorted(test),
                "warnings": [],
                "created_at": "2026-11-07T00:00:00+00:00",
                "split_protocol": "official-200 test + image-level random train/val (leaky)",
                "assignment_signature": assignment_sig(train, val, test),
                "patient_counts": {"train": len({pid(s) for s in train}), "val": len({pid(s) for s in val}), "test": len({pid(s) for s in test})},
            }
        },
    }
    (OUT / "official_leaky.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    patient = json.loads((DATA / "dataset_splits.json").read_text(encoding="utf-8"))
    (OUT / "patient_holdout.json").write_text(json.dumps(patient, ensure_ascii=False, indent=2), encoding="utf-8")

    train_pids = {pid(s) for s in patient["entries"]["holdout"]["train"]}
    clean_official = [s for s in official if pid(s) not in train_pids]
    (OUT / "clean_official_stems.txt").write_text("\n".join(sort := sorted(clean_official)) + "\n", encoding="utf-8")

    print("all", len(all_stems), "official", len(official), "leaky_train", len(train), "leaky_val", len(val))
    print("clean_official_subset", len(clean_official), "patients", len({pid(s) for s in clean_official}))
    print("official patient overlap with patient-level train", len({pid(s) for s in official} & train_pids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
