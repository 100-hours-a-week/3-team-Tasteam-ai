#!/usr/bin/env python3
"""
Adapter + base 모델을 merge하여 서빙용 단일 디렉터리로 저장.

서빙 측은 merge된 경로만 LLM_MODEL 등으로 지정하면 됨 (vLLM/풀 모델 로딩 호환).

사용:
  python scripts/merge_adapter_for_serving.py --adapter-path runs/xxx/adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --output-dir merged_models/20250101_120000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def merge_and_save(
    adapter_path: str | Path,
    base_model: str,
    output_dir: str | Path,
) -> str:
    """Base + PEFT adapter 로드 → merge_and_unload → save_pretrained. 반환: output_dir 절대 경로."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = Path(adapter_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter path not found: {adapter_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    logger.info("Loading adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    logger.info("Merging adapter into base...")
    model = model.merge_and_unload()
    logger.info("Saving merged model to: %s", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "merged_at": datetime.utcnow().isoformat() + "Z",
        "base_model": base_model,
        "adapter_path": str(adapter_path),
    }
    meta_path = output_dir / "merge_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", meta_path)
    return str(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for serving (single directory)"
    )
    parser.add_argument("--adapter-path", type=Path, required=True, help="Path to adapter (e.g. runs/xxx/adapter)")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for merged model (will be created)",
    )
    args = parser.parse_args()
    out = merge_and_save(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
    )
    print(out)


if __name__ == "__main__":
    main()
