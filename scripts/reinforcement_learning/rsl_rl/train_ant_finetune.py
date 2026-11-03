"""Run flat -> mild DR -> full DR Ant training sequentially with the stock trainer.

Example (headless, both rewards, 1000 + 500 + 500 updates per reward):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_ant_finetune.py --reward both

This launcher uses only the standard library. Every invocation starts fresh flat
training; it never picks up an unrelated/latest run. A failed stage stops the chain.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BRANCHES = {
    "original": ("", "ant_ft_original"),
    "light": ("Light-", "ant_ft_forward_light"),
}
STAGES = (("Flat", "flat"), ("Mild", "mild"), ("DR", "dr"))


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def save_manifest(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", choices=("original", "light", "both"), default="both")
    parser.add_argument("--num_envs", type=positive_int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=positive_int, nargs=3, default=(1000, 500, 500),
                        metavar=("FLAT", "MILD", "DR"), help="Additional PPO updates in each stage.")
    parser.add_argument("--device", help="Optional training device, e.g. cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training or writing files.")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("Use a fixed non-negative seed for comparable stages.")

    chain_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    manifest_path = ROOT / "logs/rsl_rl/ant_finetune_chains" / f"{chain_id}.json"
    manifest = {"chain_id": chain_id, "status": "running", "settings": vars(args), "stages": []}
    rewards = ("original", "light") if args.reward == "both" else (args.reward,)
    print(f"[CHAIN] Rewards: {', '.join(rewards)}; updates per reward: {sum(args.iterations)}", flush=True)
    if not args.dry_run:
        save_manifest(manifest_path, manifest)
        print(f"[CHAIN] Progress/checkpoints: {manifest_path}", flush=True)

    try:
        for reward in rewards:
            prefix, experiment = BRANCHES[reward]
            log_root = ROOT / "logs/rsl_rl" / experiment
            previous = None
            previous_run_pattern = None
            previous_checkpoint_name = None
            start_iteration = 0
            for index, ((task_stage, stage), iterations) in enumerate(zip(STAGES, args.iterations), 1):
                task = f"Isaac-Ant-FT-{prefix}{task_stage}-v0"
                run_name = f"auto_{chain_id}_{reward}_ft_stage{index}_{stage}"
                # The installed RSL-RL resumes numbering at the saved iteration.
                final_iteration = start_iteration + iterations - 1
                checkpoint_name = f"model_{final_iteration}.pt"
                cmd = [sys.executable, str(ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"),
                       "--task", task, "--num_envs", str(args.num_envs), "--seed", str(args.seed),
                       "--max_iterations", str(iterations), "--run_name", run_name, "--headless"]
                if args.device:
                    cmd += ["--device", args.device]
                if previous_run_pattern is not None:
                    cmd += ["--resume", "--load_run", previous_run_pattern,
                            "--checkpoint", re.escape(previous_checkpoint_name) + "$"]
                print(f"\n[CHAIN] {reward}: stage {index}/3 ({stage}), {iterations} additional updates", flush=True)
                print(shlex.join(cmd), flush=True)
                if not args.dry_run:
                    subprocess.run(cmd, cwd=ROOT, check=True)
                    runs = [p for p in log_root.glob(f"*_{run_name}") if p.is_dir()]
                    if len(runs) != 1:
                        raise RuntimeError(f"Expected exactly one new run for {run_name}; found {len(runs)}.")
                    checkpoint = runs[0] / checkpoint_name
                    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                        raise RuntimeError(f"Stage exited without its final checkpoint: {checkpoint}")
                    manifest["stages"].append({
                        "reward": reward, "stage": stage, "task": task, "iterations": iterations,
                        "resume_from": str(previous) if previous is not None else None,
                        "checkpoint": str(checkpoint),
                    })
                    save_manifest(manifest_path, manifest)
                    previous = checkpoint
                    previous_run_pattern = re.escape(runs[0].name) + "$"
                    print(f"[CHAIN] Completed: {checkpoint}", flush=True)
                else:
                    previous_run_pattern = ".*_" + re.escape(run_name) + "$"
                previous_checkpoint_name = checkpoint_name
                start_iteration = final_iteration
        manifest["status"] = "completed"
        if not args.dry_run:
            save_manifest(manifest_path, manifest)
            for record in manifest["stages"]:
                if record["stage"] == "dr":
                    print(f"[CHAIN] Final {record['reward']} checkpoint: {record['checkpoint']}", flush=True)
        print("[CHAIN] Plan complete." if args.dry_run else "[CHAIN] All requested training stages completed.", flush=True)
        return 0
    except (subprocess.CalledProcessError, OSError, RuntimeError, KeyboardInterrupt) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error) or "Keyboard interrupt"
        if not args.dry_run:
            save_manifest(manifest_path, manifest)
        print(f"[CHAIN] Stopped; no later stages will run. {manifest['error']}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
