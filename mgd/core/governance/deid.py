"""脱敏（WP-7）：影像 EXIF 清除、DICOM 身份标签清除、假名化。

合规目标：任何导出/上报的数据不得携带可识别患者或机构的明文信息。
- scrub_image_exif：像素级保留，元数据全清（EXIF / PNG text chunks 一律不带入输出）；
- scrub_dicom_tags：通用对象操作，不 import pydicom，方便对任意鸭子类型对象脱敏；
- pseudonymize：带盐哈希假名，同一患者在同一盐下保持稳定可关联，但无法反推真实 ID。
"""
from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps

#: 必须清除的患者/机构身份标签（WP-7 任务书固定清单，顺序即遍历顺序）。
DEID_TAGS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "PatientInsurancePlanCodeSequence",
    "InstitutionName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorName",
)


def scrub_image_exif(image_bytes: bytes) -> bytes:
    """清除图像元数据并统一转 PNG 字节流（保留像素）。

    流程：PIL 打开 -> ImageOps.exif_transpose（按 EXIF 方向摆正像素）-> 另存为 PNG。
    PNG 输出不携带 EXIF / PNG text chunks（PIL 重编码时不会带入源文件的元数据）。
    非图像输入抛 ValueError。
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError(f"scrub_image_exif 需要 bytes 输入，收到 {type(image_bytes).__name__}")
    try:
        with Image.open(io.BytesIO(bytes(image_bytes))) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"非图像数据，无法脱敏：{exc}") from exc
    output = io.BytesIO()
    try:
        oriented.save(output, format="PNG")
    except (OSError, ValueError) as exc:
        raise ValueError(f"图像无法以 PNG 重编码脱敏：{exc}") from exc
    return output.getvalue()


def scrub_dicom_tags(ds, mapping: dict | None = None) -> tuple[object, dict]:
    """清除通用对象（如 pydicom Dataset）上的 DEID_TAGS 身份字段。

    不 import pydicom：只做 getattr/setattr，任何带属性的对象都适用。
    - 属性存在且非空 -> mapping[tag] = str(原值)，并 setattr(ds, tag, "")；
    - 属性不存在 / 值为空（含已被脱敏为 ""）-> 跳过，不污染 mapping；
    - 传入既有 mapping 时在其上累加（便于调用方合并多次脱敏的回填记录）。
    返回 (ds, mapping)。
    """
    if mapping is None:
        mapping = {}
    for tag in DEID_TAGS:
        if not hasattr(ds, tag):
            continue
        original = getattr(ds, tag)
        if not original:
            continue
        mapping[tag] = str(original)
        setattr(ds, tag, "")
    return ds, mapping


def pseudonymize(real_id: str, salt: str) -> str:
    """带盐假名：sha256((salt + real_id).encode()) 前 16 位十六进制。

    同 (salt, real_id) 结果稳定，可跨时间关联同一患者；盐不泄露时不可反推真实 ID。
    空 real_id 或空 salt 抛 ValueError（无盐假名等于直接哈希身份，合规上禁止）。
    """
    if not isinstance(real_id, str) or not isinstance(salt, str) or not real_id or not salt:
        raise ValueError("pseudonymize 需要 non-empty 的 str real_id 与 salt")
    return hashlib.sha256((salt + real_id).encode("utf-8")).hexdigest()[:16]
