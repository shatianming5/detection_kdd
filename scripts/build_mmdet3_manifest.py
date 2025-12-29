#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty tsv: {path}")
    return rows


def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        values.append(float(raw))
    return values


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    s = f"{x:.6f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _resolve_repo_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _write_single_row_tsv(path: Path, header: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerow([row.get(k, "") for k in header])


def _update_summary(
    summary_path: Path,
    *,
    dataset: str,
    baseline_name: str,
    sample_step: int,
    ckpt_path: str,
    ckpt_sha256: str,
    results_tsv: str,
    mean_ap: float | None,
    std_ap: float | None,
    mean_mr: float | None,
    mean_ji: float | None,
    mean_fps: float | None,
) -> None:
    header = [
        "dataset",
        "baseline_name",
        "sample_step",
        "ckpt_path",
        "ckpt_sha256",
        "results_tsv",
        "mean_AP",
        "std_AP_pop",
        "mean_MR",
        "mean_JI",
        "mean_inference_fps",
    ]

    rows: list[dict[str, str]] = []
    if summary_path.is_file():
        with summary_path.open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

    def _is_same(r: dict[str, str]) -> bool:
        if (r.get("dataset") or "").strip() != dataset:
            return False
        if (r.get("baseline_name") or "").strip() != baseline_name:
            return False
        try:
            return int((r.get("sample_step") or "0").strip()) == int(sample_step)
        except Exception:
            return False

    rows = [r for r in rows if not _is_same(r)]
    rows.append(
        {
            "dataset": dataset,
            "baseline_name": baseline_name,
            "sample_step": str(int(sample_step)),
            "ckpt_path": ckpt_path,
            "ckpt_sha256": ckpt_sha256,
            "results_tsv": results_tsv,
            "mean_AP": _fmt(mean_ap),
            "std_AP_pop": _fmt(std_ap),
            "mean_MR": _fmt(mean_mr),
            "mean_JI": _fmt(mean_ji),
            "mean_inference_fps": _fmt(mean_fps),
        }
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in header])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a single-row MMDet3 manifest TSV from eval_mmdet3_multiseed TSV.")
    ap.add_argument("--metric-type", choices=["coco", "lvis", "crowdhuman"], required=True)
    ap.add_argument("--dataset", required=True, help="Dataset name for summary table (e.g. lvis/crowdhuman/coco).")
    ap.add_argument("--baseline-name", required=True)
    ap.add_argument("--results-tsv", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--summary-out", default="results/rerun_mmdet3_lvis_crowdhuman_summary.tsv")
    ap.add_argument("--update-summary", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    metric_type = str(args.metric_type)
    results_tsv_arg = str(args.results_tsv)
    results_tsv_path = _resolve_repo_path(results_tsv_arg)
    rows = _read_tsv(results_tsv_path)

    first = rows[0]
    ckpt_path = (first.get("ckpt_path") or "").strip()
    ckpt_sha256 = (first.get("ckpt_sha256") or "").strip()
    config_file = (first.get("config_file") or "").strip()
    sample_step = int((first.get("sample_step") or "0").strip())

    mean_ap, std_ap = _mean_std(_float_values(rows, "AP"))
    mean_fps, std_fps = _mean_std(_float_values(rows, "inference_fps"))
    mean_s, std_s = _mean_std(_float_values(rows, "inference_s_per_img"))

    out_row: dict[str, str] = {
        "baseline_name": str(args.baseline_name),
        "ckpt_path": ckpt_path,
        "ckpt_sha256": ckpt_sha256,
        "config_file": config_file,
        "results_tsv": results_tsv_arg,
        "sample_step": str(sample_step),
        "mean_AP": _fmt(mean_ap),
        "std_AP_pop": _fmt(std_ap),
        "mean_inference_s_per_img": _fmt(mean_s),
        "std_inference_s_per_img_pop": _fmt(std_s),
        "mean_inference_fps": _fmt(mean_fps),
        "std_inference_fps_pop": _fmt(std_fps),
    }

    if metric_type == "crowdhuman":
        mean_mr, std_mr = _mean_std(_float_values(rows, "MR"))
        mean_ji, std_ji = _mean_std(_float_values(rows, "JI"))
        header = [
            "baseline_name",
            "ckpt_path",
            "ckpt_sha256",
            "config_file",
            "results_tsv",
            "sample_step",
            "mean_AP",
            "std_AP_pop",
            "mean_MR",
            "std_MR_pop",
            "mean_JI",
            "std_JI_pop",
            "mean_inference_s_per_img",
            "std_inference_s_per_img_pop",
            "mean_inference_fps",
            "std_inference_fps_pop",
        ]
        out_row.update(
            {
                "mean_MR": _fmt(mean_mr),
                "std_MR_pop": _fmt(std_mr),
                "mean_JI": _fmt(mean_ji),
                "std_JI_pop": _fmt(std_ji),
            }
        )
        mean_mr_for_summary = mean_mr
        mean_ji_for_summary = mean_ji
    else:
        mean_ap50, std_ap50 = _mean_std(_float_values(rows, "AP50"))
        mean_ap75, std_ap75 = _mean_std(_float_values(rows, "AP75"))
        mean_aps, std_aps = _mean_std(_float_values(rows, "APs"))
        mean_apm, std_apm = _mean_std(_float_values(rows, "APm"))
        mean_apl, std_apl = _mean_std(_float_values(rows, "APl"))
        header = [
            "baseline_name",
            "ckpt_path",
            "ckpt_sha256",
            "config_file",
            "results_tsv",
            "sample_step",
            "mean_AP",
            "std_AP_pop",
            "mean_AP50",
            "std_AP50_pop",
            "mean_AP75",
            "std_AP75_pop",
            "mean_APs",
            "std_APs_pop",
            "mean_APm",
            "std_APm_pop",
            "mean_APl",
            "std_APl_pop",
            "mean_inference_s_per_img",
            "std_inference_s_per_img_pop",
            "mean_inference_fps",
            "std_inference_fps_pop",
        ]
        out_row.update(
            {
                "mean_AP50": _fmt(mean_ap50),
                "std_AP50_pop": _fmt(std_ap50),
                "mean_AP75": _fmt(mean_ap75),
                "std_AP75_pop": _fmt(std_ap75),
                "mean_APs": _fmt(mean_aps),
                "std_APs_pop": _fmt(std_aps),
                "mean_APm": _fmt(mean_apm),
                "std_APm_pop": _fmt(std_apm),
                "mean_APl": _fmt(mean_apl),
                "std_APl_pop": _fmt(std_apl),
            }
        )
        mean_mr_for_summary = None
        mean_ji_for_summary = None

    manifest_out = _resolve_repo_path(str(args.manifest_out))
    _write_single_row_tsv(manifest_out, header, out_row)

    if bool(args.update_summary):
        summary_out = _resolve_repo_path(str(args.summary_out))
        _update_summary(
            summary_out,
            dataset=str(args.dataset),
            baseline_name=str(args.baseline_name),
            sample_step=int(sample_step),
            ckpt_path=ckpt_path,
            ckpt_sha256=ckpt_sha256,
            results_tsv=results_tsv_arg,
            mean_ap=mean_ap,
            std_ap=std_ap,
            mean_mr=mean_mr_for_summary,
            mean_ji=mean_ji_for_summary,
            mean_fps=mean_fps,
        )

    print(f"wrote_manifest={manifest_out}")
    if bool(args.update_summary):
        print(f"updated_summary={_resolve_repo_path(str(args.summary_out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
