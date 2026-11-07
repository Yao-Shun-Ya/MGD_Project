"""一键自检：合成 6 例数据 → 划分 → 训练 1 epoch → 独立 Test 评估 → 报告。

用法（在仓库根或 mgd 目录均可）：
    python -m mgd.verify_pipeline --epochs 1
    python mgd/verify_pipeline.py --epochs 1

不依赖真实数据、不写仓库目录，全部落在系统临时目录。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mgd.core.config import Settings, reset_settings
    from mgd.core.data.pairing import scan_dataset
    from mgd.core.data.splits import SplitSpec, get_splits, load_or_create_splits
    from mgd.core.evaluation.evaluator import evaluate_pairs, export_csv, export_latex, summary_rows
    from mgd.core.inference.service import InferenceOptions, ModelService
    from mgd.core.reporting.report import build_report
    from mgd.core.runtime.errors import DataIntegrityError, MGDError
    from mgd.core.training.engine import TrainConfig, train
else:
    from .core.config import Settings, reset_settings
    from .core.data.pairing import scan_dataset
    from .core.data.splits import SplitSpec, get_splits, load_or_create_splits
    from .core.evaluation.evaluator import evaluate_pairs, export_csv, export_latex, summary_rows
    from .core.inference.service import InferenceOptions, ModelService
    from .core.reporting.report import build_report
    from .core.runtime.errors import DataIntegrityError, MGDError
    from .core.training.engine import TrainConfig, train


def _synthetic_pair(directory: Path, index: int) -> None:
    height, width = 320, 640
    rng = np.random.default_rng(index)
    image = np.full((height, width, 3), 36, dtype=np.uint8)
    eyelid = np.zeros((height, width), dtype=np.uint8)
    gland = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(eyelid, (width // 2, height // 2), (280, 120), 0, 0, 360, 1, -1)
    for offset in range(-90, 91, 30):
        start = (width // 2 - 230, height // 2 + offset)
        end = (width // 2 + 230, height // 2 + offset - 12)
        cv2.line(gland, start, end, 1, thickness=11)
    cv2.line(gland, (150, 120), (150, 210), 1, thickness=6)  # 一条被截断的短腺体
    noise = rng.normal(0, 6, (height, width)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + gland[:, :, None] * 120 + noise[:, :, None], 0, 255).astype(np.uint8)
    stem = f"case_{index:03d}"
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "gland_masks").mkdir(parents=True, exist_ok=True)
    (directory / "eyelid_masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "images" / f"{stem}.png"), image)
    cv2.imwrite(str(directory / "gland_masks" / f"{stem}.png"), gland * 255)
    cv2.imwrite(str(directory / "eyelid_masks" / f"{stem}.png"), eyelid * 255)


def run(samples: int = 6, epochs: int = 1, arch: str = "MGD-Former", keep: bool = False) -> int:
    root = Path(tempfile.mkdtemp(prefix="mgd_verify_"))
    data_dir = root / "data"
    for index in range(int(samples)):
        _synthetic_pair(data_dir, index)

    settings = Settings(
        home=root,
        data_dir=data_dir,
        model_dir=root / "models_weights",
        output_dir=root / "outputs",
        log_file=root / "logs" / "verify.log",
        device="cpu",
    )
    reset_settings()
    failures: list[str] = []
    print(f"[1/6] 合成数据目录：{data_dir}")
    try:
        index = scan_dataset(settings)
        print(f"      配对通过：{len(index.pairs)} 例")
    except MGDError as exc:
        print(f"      配对失败：{exc}")
        return 1

    print("[2/6] 持久化 70/15/15 划分")
    split = get_splits(index.pairs, SplitSpec(), settings)
    print(f"      train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    repeat = get_splits(index.pairs, SplitSpec(), settings)
    if [p.stem for p in repeat.train] != [p.stem for p in split.train]:
        failures.append("划分复用不一致")
    print(f"      重新加载一致：{not failures}")

    print("[3/6] 空载防火墙（<=2 例必须拒绝）")
    try:
        load_or_create_splits(index.pairs[:2], SplitSpec(), settings)
        failures.append("2 例数据未被拦截")
        print("      失败：未拦截")
    except DataIntegrityError:
        print("      通过：已拒绝")

    print(f"[4/6] 训练入口（{arch}，{epochs} epoch，CPU，tile=128）")
    config = TrainConfig(
        model_name=arch, epochs=int(epochs), batch_size=1, tile_mode=True, tile_height=128,
        val_ratio=0.15, test_ratio=0.15, seed=42, device="cpu", dual_task=True,
        loss_name="Dice+CE 组合损失", num_workers=0,
    )
    result = train(config, settings)
    print(f"      best={result.best_metric:.4f} epoch={result.best_epoch} weights={result.best_weights}")
    if result.best_weights is None:
        failures.append("训练未产出 best 权重")

    print("[5/6] 验证入口（独立 Test 集）")
    report = evaluate_pairs(
        split.test, arch=arch, settings=settings, scope="test",
        options=InferenceOptions(tile_infer=False, tta="关闭", gradcam=False),
        service=ModelService(settings, allow_unverified=True),
    )
    for row in summary_rows(report):
        print(f"      {row[0]}: {row[1]}")
    if not np.isfinite(report.summary.get("dice", float("nan"))):
        failures.append("Test Dice 非有限值")
    export_csv(report, root / "outputs" / "verify_eval.csv")
    export_latex(report, root / "outputs" / "verify_eval.tex")

    print("[6/6] 单病例 HTML 报告")
    service = ModelService(settings, allow_unverified=True)
    inference = service.predict(
        split.test[0].image, arch=arch,
        options=InferenceOptions(tile_infer=False, gradcam=True, tta="2倍增强 (水平翻转)"),
    )
    bundle = build_report(inference, model_name=arch, patient_id="verify", output_dir=root / "outputs" / "reports")
    print(f"      HTML={bundle.html_path.name} CSV={bundle.csv_path.name} PDF={bundle.pdf_path}")
    print(f"      置信度={inference.extra['confidence']['score']:.2f} 需复核={inference.extra['confidence']['need_review']}")

    print("\n=== 自检结论 ===")
    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        code = 1
    else:
        print("[PASS] 数据配对 / 划分复用 / 空载拦截 / 训练 / 评估 / 报告 全链路通过")
        code = 0
    print(f"临时目录：{root}（keep={keep}）")
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="MGD 全链路自检")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--arch", default="MGD-Former",
                        help="MGD-Former（默认）/ UNet / AttentionUNet / WRCANet / MGD-PriorNet")
    parser.add_argument("--keep", action="store_true", help="保留临时目录用于排查")
    args = parser.parse_args()
    return run(samples=args.samples, epochs=args.epochs, arch=args.arch, keep=args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
