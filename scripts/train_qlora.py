#!/usr/bin/env python3
"""
QLoRA SFT 학습 스크립트: labeled_path → adapter.

distill_strategy.md: 골드 oversample 20~30%, 총 2500~3200 학습.
kd_qlora_prefect_wandb_strategy.md: report_to=wandb, WANDB_RUN_ID=flow_run_id.

사용:
  python scripts/train_qlora.py --labeled-path labeled/xxx/train_labeled.json --student-model Qwen/Qwen2.5-0.5B-Instruct --output-dir runs/
  WANDB_RUN_ID=xxx python scripts/train_qlora.py ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# QLoRA 기본 설정
DEFAULT_R = 8
DEFAULT_ALPHA = 16
DEFAULT_TARGET_MODULES = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_NUM_EPOCHS = 2
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4


def _load_labeled(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", data) if isinstance(data, dict) else data


def _apply_oversample(samples: list[dict], gold_ratio: float) -> list[dict]:
    """골드 샘플을 oversample하여 비중을 gold_ratio에 가깝게."""
    gold = [s for s in samples if s.get("label_source") == "openai"]
    other = [s for s in samples if s.get("label_source") != "openai"]
    if not gold or gold_ratio <= 0:
        return samples
    n_other = len(other)
    n_gold_target = int(n_other * gold_ratio / (1 - gold_ratio)) if gold_ratio < 1 else len(gold)
    oversampled = gold * max(1, (n_gold_target + len(gold) - 1) // len(gold))
    oversampled = oversampled[:n_gold_target] + gold
    return oversampled + other


def _format_chat_text(instruction: str, output: str, tokenizer: Any) -> str:
    """Qwen chat 형식으로 포맷 후 apply_chat_template."""
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for distill student")
    parser.add_argument("--labeled-path", type=Path, required=True, help="train_labeled.json path")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-oversample-ratio", type=float, default=0.25)
    parser.add_argument("--r", type=int, default=DEFAULT_R)
    parser.add_argument("--alpha", type=int, default=DEFAULT_ALPHA)
    parser.add_argument("--target-modules", type=str, default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        logger.error("Required: pip install datasets transformers peft bitsandbytes trl")
        raise SystemExit(1) from e

    flow_run_id = os.environ.get("WANDB_RUN_ID") or os.environ.get("PREFECT_FLOW_RUN_ID", "")
    if flow_run_id:
        os.environ["WANDB_RUN_ID"] = flow_run_id
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_NAME"] = flow_run_id
        logger.info("wandb run_id=%s", flow_run_id)

    samples = _load_labeled(args.labeled_path)
    if args.gold_oversample_ratio > 0:
        samples = _apply_oversample(samples, args.gold_oversample_ratio)
    logger.info("Training samples: %d", len(samples))

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def fmt(ex):
        ins = ex.get("instruction", "")
        out = ex.get("output", "")
        return {"text": _format_chat_text(ins, out, tokenizer)}

    ds = Dataset.from_list([{"instruction": s["instruction"], "output": s["output"]} for s in samples])
    ds = ds.map(lambda ex: fmt(ex), remove_columns=ds.column_names, desc="formatting")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / "runs" / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(out_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=2e-5,
        max_seq_length=args.max_seq_length,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="wandb" if flow_run_id else "none",
        run_name=flow_run_id or run_id,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_path / "adapter"))
    tokenizer.save_pretrained(str(out_path / "adapter"))

    meta = {
        "labeled_path": str(args.labeled_path),
        "student_model": args.student_model,
        "n_samples": len(samples),
        "run_id": run_id,
        "adapter_path": str(out_path / "adapter"),
    }
    meta_path = out_path / "training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Saved adapter to %s", out_path / "adapter")
    print(json.dumps({"adapter_path": str(out_path / "adapter"), "training_meta_path": str(meta_path), "run_id": run_id}))


if __name__ == "__main__":
    main()
