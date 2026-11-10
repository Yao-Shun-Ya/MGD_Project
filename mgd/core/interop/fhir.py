"""FHIR R4 资源构造（WP-9）。纯 JSON，无网络、无第三方依赖。

编码占位声明（诚实优先）：本模块所有 ``code.system`` 指向 LOINC，但
``code.value``（如 ``"MGD-GA-PCT"``）是**工程占位**，不是已发布的真实
LOINC 码——严禁在临床信息学审核前对外宣称已使用正式 LOINC / SNOMED 编码。
占位值全部集中在本模块顶部的常量（``_GA_CODE`` / ``_COUNT_CODE`` /
``_SCORE_CODE`` / ``_REPORT_CODE``），上线前由临床信息学审核后一次性替换为
正式 LOINC / SNOMED CT 码。

provenance extension：url 使用占位
``https://xianeryijian.icu/fhir/StructureDefinition/provenance``
（未在 HL7 注册表登记），子扩展为 software_version / model_name /
threshold / created_at / weights_sha256（None 字段跳过）。

打包约定：:func:`to_fhir_diagnostic_report` 返回
``{"report": {...}, "observations": [...]}`` 两键 —— report 是
DiagnosticReport，observations 是其 result 引用的 Observation 列表；
不在顶层混入任何非标准键。
"""
from __future__ import annotations

import math
from typing import Any

LOINC_SYSTEM = "http://loinc.org"
UCUM_SYSTEM = "http://unitsofmeasure.org"

# 占位 URL：待正式注册后替换（见模块 docstring）
PROVENANCE_EXTENSION_URL = "https://xianeryijian.icu/fhir/StructureDefinition/provenance"

_IMAGING_CATEGORY = {
    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
    "code": "imaging",
    "display": "Imaging",
}

# 工程占位编码（非真实 LOINC 码，上线前由临床信息学审核替换，见模块 docstring）
_GA_CODE = {
    "system": LOINC_SYSTEM,
    "code": "MGD-GA-PCT",
    "display": "Meibomian gland area loss percent",
    "text": "睑板腺缺失率",
}
_COUNT_CODE = {
    "system": LOINC_SYSTEM,
    "code": "MGD-GLAND-COUNT",
    "display": "Meibomian gland count",
    "text": "睑板腺腺体计数",
}
_SCORE_CODE = {
    "system": LOINC_SYSTEM,
    "code": "MGD-MEIBOSCORE",
    "display": "Meibomian gland meiboscore",
    "text": "Meiboscore 评分",
}
_REPORT_CODE = {
    "system": LOINC_SYSTEM,
    "code": "MGD-REPORT",
    "display": "Meibomian gland dysfunction analysis report",
    "text": "睑板腺缺失分析报告",
}

# 汇总标量 → (Observation id, code, 单位, JSON 数值类型)
_SCALAR_FIELDS = (
    ("ga_percent", "mgd-ga-percent", _GA_CODE, "%", float),
    ("instance_count", "mgd-instance-count", _COUNT_CODE, "1", int),
    ("meiboscore", "mgd-meiboscore", _SCORE_CODE, "1", int),
)


def _slug(key: str) -> str:
    """把任意字段名压成 id / 占位 code 可用的 ASCII 短标识。"""
    cleaned = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_"
        for ch in str(key)
    ).strip("_")
    return (cleaned or "field")


def _require_subject_ref(subject_ref: Any) -> None:
    if not isinstance(subject_ref, str) or not subject_ref.strip():
        raise ValueError("subject_ref 必须是非空字符串（FHIR 相对引用，如 'Patient/1'）")


def _finite_number(value: Any, name: str) -> float:
    """转 float 并拒绝 NaN / Inf：FHIR JSON 不允许输出非有限数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值，实际为 {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(
            f"{name} 不允许为 NaN / Inf（FHIR JSON 禁止输出非有限数）"
        )
    return number


def _provenance_extension(provenance: Any) -> list[dict]:
    """把 Provenance 转成占位 url 的 extension（None 字段跳过；空则返回 []）。"""
    if provenance is None:
        return []
    parts: list[dict] = []
    software_version = getattr(provenance, "software_version", None)
    if software_version:
        parts.append({"url": "software_version", "valueString": str(software_version)})
    model_name = getattr(provenance, "model_name", None)
    if model_name:
        parts.append({"url": "model_name", "valueString": str(model_name)})
    threshold = getattr(provenance, "threshold", None)
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = None
        if threshold is not None and math.isfinite(threshold):
            parts.append({"url": "threshold", "valueDecimal": threshold})
    created_at = getattr(provenance, "created_at", None)
    if created_at:
        parts.append({"url": "created_at", "valueDateTime": str(created_at)})
    weights_sha256 = getattr(provenance, "weights_sha256", None)
    if weights_sha256:
        parts.append({"url": "weights_sha256", "valueString": str(weights_sha256)})
    if not parts:
        return []
    return [{"url": PROVENANCE_EXTENSION_URL, "extension": parts}]


def _build_observation(
    *,
    obs_id: str,
    code: dict,
    value: float,
    unit: str,
    subject_ref: str,
    provenance: Any,
) -> dict:
    """单个 Observation 的公共骨架（code 为工程占位编码，见模块 docstring）。"""
    _require_subject_ref(subject_ref)
    observation: dict = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [{"coding": [dict(_IMAGING_CATEGORY)]}],
        "code": {"coding": [dict(code)], "text": code["text"]},
        "subject": {"reference": subject_ref},
        "valueQuantity": {
            "value": value,
            "unit": unit,
            "system": UCUM_SYSTEM,
            "code": unit,
        },
    }
    if provenance is not None:
        created_at = getattr(provenance, "created_at", None)
        if created_at:
            observation["effectiveDateTime"] = str(created_at)
        extension = _provenance_extension(provenance)
        if extension:
            observation["extension"] = extension
    return observation


def to_fhir_observation(measurement: dict, subject_ref: str, provenance: Any = None) -> dict:
    """把单条测量构造成 FHIR Observation（当前针对腺体缺失率）。

    参数：
        measurement: 至少含 ``ga_percent``（数值，缺失 → ValueError；
            NaN / Inf → ValueError，绝不允许输出 NaN JSON）。
            可用 ``measurement["unit"]`` 覆盖默认单位 ``"%"``（UCUM）。
        subject_ref: FHIR 相对引用（如 ``"Patient/1"``）。
        provenance: :class:`mgd.core.provenance.Provenance`；其 ``created_at``
            → ``effectiveDateTime``，其余字段 → 占位 extension（url 见
            :data:`PROVENANCE_EXTENSION_URL`，未注册占位）。provenance 为
            None 时不写 effectiveDateTime（FHIR 允许缺省，不写 null）。

    返回：
        JSON 可序列化的 Observation dict（纯标准库类型，可直接 json.dumps）。
    """
    if not isinstance(measurement, dict):
        raise ValueError("measurement 必须是 dict")
    if "ga_percent" not in measurement:
        raise ValueError("measurement 缺少必需字段: ['ga_percent']")
    value = _finite_number(measurement["ga_percent"], "ga_percent")
    unit = str(measurement.get("unit") or "%")
    return _build_observation(
        obs_id="mgd-ga-percent",
        code=_GA_CODE,
        value=value,
        unit=unit,
        subject_ref=subject_ref,
        provenance=provenance,
    )


def to_fhir_diagnostic_report(
    measurements: dict, subject_ref: str, provenance: Any = None
) -> dict:
    """把整份测量汇总构造成 FHIR DiagnosticReport + Observation 列表。

    打包约定：返回 ``{"report": {...}, "observations": [...]}`` 两键
    （docstring 约定，不把非标准键混进顶层）：

    - ``observations``：measurements 的每个**数值标量**（int/float，不含
      bool；NaN / Inf → ValueError）生成一个 Observation，
      ``len(observations) == 标量字段数``；已知键（ga_percent /
      instance_count / meiboscore）用占位编码与 UCUM 单位（``%`` / ``1`` /
      ``1``），其余数值键用 ``MGD-<SLUG>`` 占位码 + 无量纲单位；
    - ``report``：DiagnosticReport，``result`` 指向上述 Observation 的
      占位相对引用（``Observation/<id>``），``conclusion`` 为中文摘要，
      ``presentedForm`` 固定空数组（暂不内嵌文件）。

    ``measurements`` 缺 ``ga_percent`` → ValueError（消息列出缺失 key）。

    provenance 非 None 时：report 与各 Observation 的 ``effectiveDateTime``
    取 ``provenance.created_at``，并附占位 provenance extension。
    """
    if not isinstance(measurements, dict):
        raise ValueError("measurements 必须是 dict")
    if "ga_percent" not in measurements:
        raise ValueError("measurements 缺少必需字段: ['ga_percent']")
    _require_subject_ref(subject_ref)

    observations: list[dict] = []
    result: list[dict] = []
    used: set[str] = set()
    for key, obs_id, code, unit, caster in _SCALAR_FIELDS:
        if key not in measurements:
            continue
        number = _finite_number(measurements[key], key)
        observations.append(
            _build_observation(
                obs_id=obs_id,
                code=code,
                value=caster(number),
                unit=unit,
                subject_ref=subject_ref,
                provenance=provenance,
            )
        )
        result.append({"reference": f"Observation/{obs_id}"})
        used.add(key)
    for key, value in measurements.items():
        if key in used:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue  # 字符串 / 列表 / 字典等非数值标量不生成 Observation
        number = _finite_number(value, key)
        slug = _slug(key)
        obs_id = f"mgd-{slug.lower()}"
        code = {
            "system": LOINC_SYSTEM,
            "code": f"MGD-{slug.upper()}",
            "display": str(key),
            "text": str(key),
        }
        observations.append(
            _build_observation(
                obs_id=obs_id,
                code=code,
                value=number,
                unit="1",
                subject_ref=subject_ref,
                provenance=provenance,
            )
        )
        result.append({"reference": f"Observation/{obs_id}"})

    # 中文摘要（缺失的次要指标自动略去）
    ga_percent = _finite_number(measurements["ga_percent"], "ga_percent")
    summary_parts = [f"睑板腺缺失率 {ga_percent:.2f}%"]
    for key, label, fmt in (
        ("instance_count", "腺体计数", "{:.0f}"),
        ("meiboscore", "Meiboscore", "{:.0f}"),
    ):
        if key in measurements:
            number = _finite_number(measurements[key], key)
            summary_parts.append(f"{label} {fmt.format(number)}")
    conclusion = "，".join(summary_parts) + "。"

    report: dict = {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "code": {"coding": [dict(_REPORT_CODE)], "text": _REPORT_CODE["text"]},
        "subject": {"reference": subject_ref},
        "result": result,
        "conclusion": conclusion,
        "presentedForm": [],
    }
    if provenance is not None:
        created_at = getattr(provenance, "created_at", None)
        if created_at:
            report["effectiveDateTime"] = str(created_at)
        extension = _provenance_extension(provenance)
        if extension:
            report["extension"] = extension
    return {"report": report, "observations": observations}
