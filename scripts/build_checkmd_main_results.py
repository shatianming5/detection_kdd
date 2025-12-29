#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("generated_at_utc"):
            raise ValueError(f"Unexpected manifest header: {header.strip()}")
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            exp_id = row.get("exp_id", "").strip()
            if not exp_id:
                continue
            rows[exp_id] = row
    return rows


def summarize_tsv(tsv_path: Path) -> Dict[str, float]:
    with tsv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        ap: List[float] = []
        ap75: List[float] = []
        fps: List[float] = []
        for row in r:
            if "AP" in row and row["AP"]:
                ap.append(float(row["AP"]))
            if "AP75" in row and row["AP75"]:
                ap75.append(float(row["AP75"]))
            if "inference_fps" in row and row["inference_fps"]:
                fps.append(float(row["inference_fps"]))

    out: Dict[str, float] = {}
    if ap:
        out["AP_mean"] = float(statistics.mean(ap))
        out["AP_std_pop"] = float(statistics.pstdev(ap)) if len(ap) > 1 else 0.0
    if ap75:
        out["AP75_mean"] = float(statistics.mean(ap75))
        out["AP75_std_pop"] = float(statistics.pstdev(ap75)) if len(ap75) > 1 else 0.0
    if fps:
        out["FPS_mean"] = float(statistics.mean(fps))
        out["FPS_std_pop"] = float(statistics.pstdev(fps)) if len(fps) > 1 else 0.0
    out["n_runs"] = float(len(ap))
    return out


def format_float(x: float | None, *, ndigits: int = 2) -> str:
    if x is None:
        return ""
    return f"{x:.{ndigits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results_manifest.tsv")
    ap.add_argument("--spec", required=True, help="JSON file listing table rows")
    ap.add_argument("--out", default="", help="Output markdown path (default: stdout)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    spec_path = Path(args.spec)
    manifest = read_manifest(manifest_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = spec.get("rows", spec)
    if not isinstance(rows, list):
        raise ValueError("Spec must be a JSON list or an object with `rows: [...]`")

    table_rows: List[Tuple[str, str, str, str, str, str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Each row in spec must be an object")
        method = str(item.get("method", ""))
        mtype = str(item.get("type", ""))
        backbone = str(item.get("backbone", ""))
        epochs = str(item.get("epochs", ""))
        exp_id = str(item.get("exp_id", "")).strip()
        notes = str(item.get("notes", ""))

        ap_mean = ap75_mean = fps_mean = None
        if exp_id and exp_id in manifest:
            tsv_rel = manifest[exp_id].get("tsv_path", "").strip()
            if tsv_rel:
                tsv_path = (manifest_path.parent / tsv_rel).resolve()
                if tsv_path.exists():
                    stats = summarize_tsv(tsv_path)
                    ap_mean = stats.get("AP_mean")
                    ap75_mean = stats.get("AP75_mean")
                    fps_mean = stats.get("FPS_mean")
                else:
                    notes = (notes + f" (missing tsv: {tsv_rel})").strip()
        elif exp_id:
            notes = (notes + " (exp_id not in manifest)").strip()

        table_rows.append(
            (
                method,
                mtype,
                backbone,
                epochs,
                format_float(ap_mean, ndigits=2),
                format_float(ap75_mean, ndigits=2),
                format_float(fps_mean, ndigits=1),
                notes,
            )
        )

    md_lines: List[str] = []
    md_lines.append("| Method | Type | Backbone | Epochs | AP | AP75 | FPS | Notes |")
    md_lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for r in table_rows:
        md_lines.append("| " + " | ".join(r) + " |")
    md = "\n".join(md_lines) + "\n"

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md, end="")


if __name__ == "__main__":
    main()

