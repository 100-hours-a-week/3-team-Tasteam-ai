#!/usr/bin/env python3
"""
Pod에서 실행: eval_distill.py 실행 후 결과 디렉터리를 wandb artifact로 업로드하고,
완료 정보를 eval_done.json에 기록한다.
로컬에서는 eval_done.json의 qualified_name으로 artifact를 다운로드할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval_distill then upload result to wandb artifact")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--val-labeled", type=Path, default=None)
    parser.add_argument("--test-labeled", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True, help="eval_distill --output-dir (eval/<run_id> 생성됨)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--artifact-name", type=str, default="distill-eval-report")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--max-eval", type=int, default=0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "eval_distill.py"),
        "--adapter-path", str(args.adapter_path),
        "--base-model", args.base_model,
        "--output-dir", str(args.output_dir),
    ]
    if args.val_labeled and args.val_labeled.exists():
        cmd.extend(["--val-labeled", str(args.val_labeled)])
    if args.test_labeled and args.test_labeled.exists():
        cmd.extend(["--test-labeled", str(args.test_labeled)])
    if args.max_eval > 0:
        cmd.extend(["--max-eval", str(args.max_eval)])

    logger.info("Running eval_distill: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_SCRIPT_DIR.parent))
    if result.returncode != 0:
        logger.error("eval_distill stderr: %s", result.stderr)
        raise RuntimeError(f"eval_distill.py exited with {result.returncode}")

    # 마지막 JSON 줄에서 report_path 추출
    report_path = None
    for line in reversed((result.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                out = json.loads(line)
                report_path = out.get("report_path")
                break
            except json.JSONDecodeError:
                continue
    if not report_path:
        raise RuntimeError("eval_distill.py did not print report_path JSON")

    eval_dir = Path(report_path).resolve().parent
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    # wandb artifact 업로드
    qualified_name = None
    version = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            project = args.wandb_project or os.environ.get("WANDB_PROJECT", "tasteam-distill")
            entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or ""
            run = wandb.init(project=project, entity=entity or None, job_type="eval_upload")
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type="eval-report",
                metadata={
                    "adapter_path": str(args.adapter_path),
                    "base_model": args.base_model,
                    "report_path": report_path,
                },
            )
            artifact.add_dir(str(eval_dir), name="eval")
            wandb.log_artifact(artifact)
            # 동일 이름 재업로드 시 wandb가 v0, v1, ... 부여. qualified_name으로 다운로드 시 버전 지정 가능
            version = getattr(artifact, "version", None) or "latest"
            qualified_name = f"{run.entity}/{run.project}/{args.artifact_name}:{version}"
            wandb.finish()
            logger.info("Uploaded artifact %s", qualified_name)
        except Exception as e:
            logger.warning("wandb artifact upload failed: %s", e)
            qualified_name = None
            version = None
    else:
        logger.warning("WANDB_API_KEY not set; skipping artifact upload")

    # 완료 마커 기록 (로컬에서 폴링 후 artifact 다운로드용)
    done_path = args.output_dir / "eval_done.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = {
        "report_path": report_path,
        "eval_dir": str(eval_dir),
        "artifact_name": args.artifact_name,
        "version": version,
        "qualified_name": qualified_name,
    }
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", done_path)
    print(json.dumps(done, ensure_ascii=False))


if __name__ == "__main__":
    main()
