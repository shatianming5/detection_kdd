#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
import signal
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_env_kv(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set-env expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"invalid env key in {item!r}")
        env[k] = v
    return env


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"[{_now()}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _maybe_pause(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), signal.SIGSTOP)
    except ProcessLookupError:
        print(f"[{_now()}] warn: pause_pid_not_found pid={pid}", flush=True)
        return False
    print(f"[{_now()}] paused pid={pid}", flush=True)
    return True


def _maybe_resume(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(int(pid), signal.SIGCONT)
    except ProcessLookupError:
        print(f"[{_now()}] warn: resume_pid_not_found pid={pid}", flush=True)
        return
    print(f"[{_now()}] resumed pid={pid}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for a checkpoint then run MMDet3 multi-seed eval + manifest.")
    ap.add_argument("--dataset", required=True, help="Dataset name for summary (e.g. lvis/crowdhuman/coco).")
    ap.add_argument("--metric-type", choices=["coco", "lvis", "crowdhuman"], required=True)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-tag", required=True, help="Used to name TSV/manifest files and eval exp_name.")
    ap.add_argument("--baseline-prefix", required=True, help="Used to name manifest baseline_name (prefix + _stepX).")
    ap.add_argument("--out-base", default="/data/tiasha_archives_runs/mmdet3_evals")
    ap.add_argument("--summary-out", default="rerun_mmdet3_lvis_crowdhuman_summary.tsv")
    ap.add_argument("--steps", nargs="+", type=int, default=[10, 1])
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--conda-env", default="mmdet3")
    ap.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wait-interval-s", type=int, default=600)
    ap.add_argument("--set-env", action="append", default=[], help="Extra env vars to pass to eval (KEY=VALUE).")
    ap.add_argument(
        "--pause-train-pid",
        type=int,
        default=None,
        help="If set, send SIGSTOP to this PID before eval, then SIGCONT after. "
        "Useful when training and eval share a single GPU.",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if bool(args.wait):
        while not ckpt_path.is_file():
            print(f"[{_now()}] waiting_for_ckpt={ckpt_path}", flush=True)
            time.sleep(max(int(args.wait_interval_s), 1))
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    env = os.environ.copy()
    env.update(_parse_env_kv(list(args.set_env)))

    paused = _maybe_pause(args.pause_train_pid)
    try:
        for step in args.steps:
            exp_name = f"{args.run_tag}_step{int(step)}"
            tsv_out = f"{exp_name}_results.tsv"
            manifest_out = f"{exp_name}_manifest.tsv"
            baseline_name = f"{args.baseline_prefix}_step{int(step)}"

            _run(
                [
                    "python",
                    "scripts/eval_mmdet3_multiseed.py",
                    "--exp-name",
                    exp_name,
                    "--config-file",
                    str(args.config_file),
                    "--checkpoint",
                    str(ckpt_path),
                    "--tsv-out",
                    tsv_out,
                    "--out-base",
                    str(args.out_base),
                    "--metric-type",
                    str(args.metric_type),
                    "--sample-step",
                    str(int(step)),
                    "--conda-env",
                    str(args.conda_env),
                    "--resume",
                    "--eval-seeds",
                    *[str(int(s)) for s in args.eval_seeds],
                ],
                env=env,
            )

            _run(
                [
                    "python",
                    "scripts/build_mmdet3_manifest.py",
                    "--metric-type",
                    str(args.metric_type),
                    "--dataset",
                    str(args.dataset),
                    "--baseline-name",
                    baseline_name,
                    "--results-tsv",
                    tsv_out,
                    "--manifest-out",
                    manifest_out,
                    "--summary-out",
                    str(args.summary_out),
                    "--update-summary",
                ],
                env=env,
            )
    finally:
        if paused:
            _maybe_resume(args.pause_train_pid)

    print(f"[{_now()}] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
