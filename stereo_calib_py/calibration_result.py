"""Write release calibration JSON from the full BA result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .project import PROJECT_ROOT


DEFAULT_TEMPLATE = PROJECT_ROOT / "templates/calibration_result_template.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_calibration_result(ba_result: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    """Fill the release result template with the final BA calibration values."""
    for section in ("left", "right"):
        for name in template[section]:
            template[section][name] = ba_result[section][name]
    template["extrinsics"]["R"] = ba_result["extrinsics"]["R"]
    template["extrinsics"]["t"] = ba_result["extrinsics"]["t"]
    return template


def write_calibration_result(
    ba_result_json: Path,
    output_json: Path,
    template_json: Path = DEFAULT_TEMPLATE,
) -> None:
    """Write only final calibration parameters to ``output_json``."""
    if not ba_result_json.exists():
        raise FileNotFoundError(f"BA result JSON was not generated: {ba_result_json}")
    ba_result = _load_json(ba_result_json)
    template = _load_json(template_json)
    calibration = build_calibration_result(ba_result, template)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
        f.write("\n")
