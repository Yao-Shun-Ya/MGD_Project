"""预处理口径：形状、归一化范围、训练与推理逐像素一致。"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from mgd.core.data.pairing import build_data_dicts, scan_dataset
from mgd.core.data.transforms import build_eval_transforms, build_train_transforms
from mgd.core.imaging.preprocess import (
    apply_clahe,
    load_gray,
    preprocess_array,
    preprocess_path,
    resize_gray,
    to_gray,
)


def test_preprocess_shape_dtype_range(dataset_builder):
    settings = dataset_builder()
    index = scan_dataset(settings)
    pre = preprocess_path(index.pairs[0].image, settings)
    assert pre.resized.shape == settings.target_size
    assert pre.resized.dtype == np.uint8
    assert pre.model_input.shape == settings.target_size
    assert pre.model_input.dtype == np.float32
    assert 0.0 <= float(pre.model_input.min()) and float(pre.model_input.max()) <= 1.0


def test_training_and_inference_preprocess_are_pixel_identical(dataset_builder):
    settings = dataset_builder()
    index = scan_dataset(settings)
    data = build_data_dicts(index.pairs[:1])
    transformed = build_eval_transforms(settings=settings)(data[0])

    reference = preprocess_path(index.pairs[0].image, settings)
    trained_image = transformed["image"].numpy()
    assert trained_image.shape == (1, *settings.target_size)
    assert trained_image.dtype == np.float32
    assert np.allclose(trained_image[0], reference.model_input, atol=1e-6), "训练与推理预处理必须逐像素一致"


def test_labels_are_binarized_and_shape_preserved(dataset_builder):
    settings = dataset_builder()
    index = scan_dataset(settings)
    transformed = build_eval_transforms(settings=settings)(build_data_dicts(index.pairs[:1])[0])
    for key in ("label", "eyelid"):
        value = transformed[key]
        assert value.shape == (1, *settings.target_size)
        assert torch.all((value == 0) | (value == 1))


def test_mask_pixel_count_is_preserved_without_resize(dataset_builder):
    settings = dataset_builder()
    index = scan_dataset(settings)
    transformed = build_eval_transforms(settings=settings)(build_data_dicts(index.pairs[:1])[0])
    original = np.asarray(Image.open(index.pairs[0].gland_mask))
    assert int(transformed["label"].sum()) == int((original > 0).sum())


def test_tile_mode_output_size(dataset_builder):
    settings = dataset_builder()
    index = scan_dataset(settings)
    transform = build_train_transforms(
        tile_mode=True, tile_height=32, augmentation=False, settings=settings
    )
    for _ in range(3):
        out = transform(build_data_dicts(index.pairs[:1])[0])
        assert out["image"].shape == (1, 32, 64)
        assert out["label"].shape == (1, 32, 64)


def test_rgb_and_gray_inputs_are_equivalent(dataset_builder):
    settings = dataset_builder()
    index = scan_dataset(settings)
    gray = load_gray(index.pairs[0].image)
    from_gray = preprocess_array(gray, settings).model_input
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    assert np.array_equal(to_gray(rgb), gray)
    from_rgb = preprocess_array(rgb, settings).model_input
    assert np.allclose(from_gray, from_rgb, atol=1e-6)


def test_float_and_uint8_gray_conversion():
    values = np.arange(64 * 128, dtype=np.uint8).reshape(64, 128)
    assert np.array_equal(to_gray(values), values)
    as_float = values.astype(np.float32) / 255.0
    converted = to_gray(as_float)
    assert converted.dtype == np.uint8
    assert np.abs(converted.astype(int) - values.astype(int)).max() <= 1


def test_resize_is_identity_at_target_size():
    image = np.zeros((64, 128), dtype=np.uint8)
    image[10:20, 10:20] = 255
    assert np.array_equal(resize_gray(image, (64, 128)), image)


def test_clahe_output_contract():
    image = np.random.default_rng(0).integers(0, 256, (64, 128), dtype=np.uint8)
    enhanced = apply_clahe(image)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == image.shape
    assert 0.0 <= enhanced.min() and enhanced.max() <= 1.0


def test_unsupported_shape_raises():
    with pytest.raises(ValueError):
        to_gray(np.zeros((4, 4, 5), dtype=np.uint8))
