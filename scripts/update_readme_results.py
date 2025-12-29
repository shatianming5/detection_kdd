#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fmt_float(raw: str, *, ndigits: int = 2) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        return f"{float(raw):.{ndigits}f}"
    except Exception:
        return raw


def _read_summary(summary_tsv: Path) -> list[dict[str, str]]:
    with summary_tsv.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows


def _build_mmdet3_table_rows(summary_rows: list[dict[str, str]]) -> list[list[str]]:
    out: list[list[str]] = []
    for r in summary_rows:
        dataset = (r.get("dataset") or "").strip()
        baseline_name = (r.get("baseline_name") or "").strip()
        step = (r.get("sample_step") or "").strip()
        mean_ap = _fmt_float(r.get("mean_AP") or "", ndigits=2)
        std_ap = _fmt_float(r.get("std_AP_pop") or "", ndigits=2)
        mean_fps = _fmt_float(r.get("mean_inference_fps") or "", ndigits=2)
        ckpt = Path((r.get("ckpt_path") or "").strip()).name
        out.append([dataset, baseline_name, step, mean_ap, std_ap, mean_fps, ckpt])

    def _key(row: list[str]) -> tuple:
        # Dataset, then baseline, then step (numeric if possible).
        try:
            step_i = int(row[2])
        except Exception:
            step_i = 0
        return (row[0], row[1], step_i)

    out.sort(key=_key)
    return out


def _render_md_table(rows: list[list[str]]) -> list[str]:
    header = ["Dataset", "baseline_name", "step", "mean_AP", "std_AP", "mean_fps", "ckpt"]
    md: list[str] = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                ]
            )
            + " |"
        )
    return md


def _replace_md_table(readme_lines: list[str], table_header_prefix: str, new_table_lines: list[str]) -> list[str]:
    start = None
    for i, line in enumerate(readme_lines):
        if line.strip() == table_header_prefix.strip():
            start = i
            break
    if start is None:
        raise RuntimeError(f"Could not find table header line in README: {table_header_prefix!r}")

    end = start + 1
    while end < len(readme_lines) and readme_lines[end].lstrip().startswith("|"):
        end += 1

    return readme_lines[:start] + new_table_lines + readme_lines[end:]


def main() -> int:
    ap = argparse.ArgumentParser(description="Update README MMDet3 results table from summary TSV.")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--summary", default="results/rerun_mmdet3_lvis_crowdhuman_summary.tsv")
    args = ap.parse_args()

    readme_path = (REPO_ROOT / args.readme).resolve()
    summary_path = (REPO_ROOT / args.summary).resolve()

    if not readme_path.is_file():
        raise FileNotFoundError(readme_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary_rows = _read_summary(summary_path)
    table_rows = _build_mmdet3_table_rows(summary_rows)
    new_table = _render_md_table(table_rows)

    lines = readme_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    updated = _replace_md_table(
        lines,
        "| Dataset | baseline_name | step | mean_AP | std_AP | mean_fps | ckpt |",
        new_table,
    )
    readme_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

