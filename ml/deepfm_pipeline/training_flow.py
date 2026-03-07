"""
Prefect 기반 DeepFM 학습 파이프라인 (deepfm_design.md §6-2).

- (선택) train/test 분할 → 전처리 → 학습 → 모델 저장 + pipeline_version 발급 → 오프라인 지표 기록

실행 (프로젝트 루트에서):
  python ml/deepfm_pipeline/training_flow.py

또는 deepfm_pipeline 디렉터리에서:
  python training_flow.py

source_dataset_path를 주면 해당 CSV를 train.csv/test.csv로 나눈 뒤 전처리·학습.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# deepfm_training 루트를 path에 넣어 model/data/utils import 가능하게 함 (repo root에서 실행 시)
_deepfm_root = Path(__file__).resolve().parent
if str(_deepfm_root) not in sys.path:
    sys.path.insert(0, str(_deepfm_root))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from prefect import flow, task
from prefect.context import get_run_context

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset


# --- 기본 경로 (deepfm_training 기준) ---
def _default_raw_data_dir() -> str:
    return str(_deepfm_root / "data" / "raw")


def _default_processed_data_dir() -> str:
    return str(_deepfm_root / "data")


def _default_output_dir() -> str:
    return str(_deepfm_root / "output")


def _generate_pipeline_version() -> str:
    """deepfm_design §6-2: pipeline_version 발급 (예: deepfm-1.0.20260227120000)."""
    now = datetime.now(timezone.utc)
    return f"deepfm-1.0.{now.strftime('%Y%m%d%H%M%S')}"


@task(name="deepfm-split-train-test", log_prints=True)
def split_train_test_task(
    source_dataset_path: str,
    raw_data_dir: str,
    test_ratio: float = 0.2,
    random_state: int | None = 42,
    time_column: str | None = None,
) -> str:
    """
    단일 CSV를 train.csv / test.csv로 나눠 raw_data_dir에 저장.
    - time_column이 있으면 시간 기준으로 정렬 후 뒤쪽 test_ratio를 test로 둠.
    - time_column이 없으면 shuffle 후 뒤쪽 test_ratio를 test로 둠.
    """
    import pandas as pd

    path = Path(source_dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset_path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Source dataset is empty: {source_dataset_path}")

    n = len(df)
    test_size = max(1, int(n * test_ratio))
    train_size = n - test_size

    if time_column and time_column in df.columns:
        df = df.sort_values(time_column, na_position="last").reset_index(drop=True)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
    else:
        if random_state is not None:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

    raw = Path(raw_data_dir)
    raw.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(raw / "train.csv", index=False)
    test_df.to_csv(raw / "test.csv", index=False)
    print(f"Split done: train={len(train_df)}, test={len(test_df)} -> {raw_data_dir}")
    return raw_data_dir


@task(name="deepfm-preprocess", log_prints=True)
def preprocess_task(
    raw_data_dir: str,
    processed_data_dir: str,
    num_train_sample: int | None = 9000,
    num_test_sample: int | None = 1000,
    use_sample_weight: bool = True,
    time_column: str | None = None,
    train_end: str | None = None,
    valid_end: str | None = None,
    test_end: str | None = None,
    group_column: str | None = None,
    negative_sampling_ratio: float = 1.0,
    negative_sampling_seed: int = 42,
    eval_list_size: int = 101,
    eval_num_neg: int = 100,
    eval_num_popular_neg: int = 50,
    eval_popular_top_k: int = 1000,
    eval_list_seed: int = 42,
    use_wandb: bool = True,
    exp_ablation: list[str] | None = None,
) -> str:
    """
    전처리 + (wandb_design) split 메타·feature_sizes·dataset 스냅샷 artifact 로깅.
    eval_list_size>0이면 test/val을 리스트당 1 pos + eval_num_neg neg로 재구성.
    exp_ablation: exp 피처 ablation (예: ["user_category_match","user_region_match","price_diff"]).
    """
    from utils.dataPreprocess import preprocess
    from utils import wandb_logger

    raw = Path(raw_data_dir)
    out = Path(processed_data_dir)
    if not raw.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    out.mkdir(parents=True, exist_ok=True)

    preprocess(
        datadir=str(raw),
        outdir=str(out),
        num_train_sample=num_train_sample,
        num_test_sample=num_test_sample,
        use_sample_weight=use_sample_weight,
        time_column=time_column,
        train_end=train_end,
        valid_end=valid_end,
        test_end=test_end,
        group_column=group_column,
        negative_sampling_ratio=negative_sampling_ratio,
        negative_sampling_seed=negative_sampling_seed,
        eval_list_size=eval_list_size,
        eval_num_neg=eval_num_neg,
        eval_num_popular_neg=eval_num_popular_neg,
        eval_popular_top_k=eval_popular_top_k,
        eval_list_seed=eval_list_seed,
        exp_ablation=exp_ablation,
    )
    print(f"Preprocess done: {processed_data_dir}")

    if use_wandb and wandb_logger.is_available():
        stats = wandb_logger.dataset_stats_from_processed_dir(out)
        (out / "dataset_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if (out / "split_meta.json").exists():
            wandb_logger.log_artifact("split_metadata", "split_metadata", out / "split_meta.json")
        if (out / "feature_sizes.txt").exists():
            wandb_logger.log_artifact("feature_sizes", "feature_sizes", out / "feature_sizes.txt")
        wandb_logger.log_artifact("dataset_stats", "dataset_stats", out / "dataset_stats.json")
    return processed_data_dir


@task(name="deepfm-train", log_prints=True)
def train_task(
    processed_data_dir: str,
    num_train: int = 9000,
    num_val: int = 1000,
    epochs: int = 5,
    batch_size: int = 100,
    lr: float = 1e-4,
    output_dir: str | None = None,
    use_cuda: bool = False,
    verbose: bool = True,
    use_wandb: bool = True,
    seed: int = 42,
) -> dict:
    """
    전처리된 데이터로 DeepFM 학습 후 모델 저장.
    seed 고정 시 동일 실행마다 동일 결과 (모델 초기화·배치 셔플 재현).
    (wandb_design) checkpoint·evaluation report artifact, metrics 로깅, pipeline_version 매핑.
    """
    from utils import wandb_logger

    # 재현성: 모델 초기화·DataLoader 셔플 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    data_path = Path(processed_data_dir)
    if not (data_path / "train.txt").exists():
        raise FileNotFoundError(f"Processed train.txt not found in {processed_data_dir}")

    out_path = Path(output_dir or _default_output_dir())
    run_ctx = get_run_context()
    run_id = getattr(run_ctx, "flow_run", None) and getattr(run_ctx.flow_run, "id", None)
    run_name = str(run_id) if run_id else "manual"
    out_run = out_path / run_name
    out_run.mkdir(parents=True, exist_ok=True)

    train_config = {
        "num_train": num_train,
        "num_val": num_val,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "use_cuda": use_cuda,
        "seed": seed,
    }

    # 데이터 로드: 시간 기준 split 시 val.txt 사용, 없으면 train 뒤쪽을 val로 (랜덤 row split 아님)
    train_data = CriteoDataset(str(data_path), train=True)
    val_path = data_path / "val.txt"
    if val_path.exists():
        val_data = CriteoDataset(str(data_path), train=False, use_val_file=True)
        loader_train = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, generator=g
        )
        loader_val = DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )
    else:
        num_total = len(train_data)
        val_start = max(0, num_total - num_val)
        train_indices = list(range(0, val_start))
        val_indices = list(range(val_start, num_total))
        loader_train = DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=sampler.SubsetRandomSampler(train_indices),
            generator=g,
        )
        loader_val = DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=sampler.SubsetRandomSampler(val_indices),
            generator=g,
        )

    feature_sizes_path = data_path / "feature_sizes.txt"
    feature_sizes = np.loadtxt(str(feature_sizes_path), delimiter=",")
    feature_sizes = [int(x) for x in feature_sizes]

    model = DeepFM(feature_sizes, use_cuda=use_cuda)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    model.fit(loader_train, loader_val, optimizer, epochs=epochs, verbose=verbose)

    # §6-2: pipeline_version 발급
    pipeline_version = _generate_pipeline_version()
    version_path = out_run / "pipeline_version.txt"
    version_path.write_text(pipeline_version, encoding="utf-8")

    # 모델 및 메타 저장
    model_path = out_run / "model.pt"
    torch.save(model.state_dict(), model_path)
    meta_path = out_run / "feature_sizes.txt"
    np.savetxt(str(meta_path), feature_sizes, fmt="%d", delimiter=",")
    print(f"Model saved: {model_path}, pipeline_version: {pipeline_version}")

    # §6-2: 오프라인 지표 산출(NDCG@K / Recall@K / AUC) + popularity baseline 기록
    run_metrics = {}
    test_path = data_path / "test.txt"
    if test_path.exists():
        from utils.evaluate import run_evaluation, run_popularity_baseline, run_random_baseline
        run_metrics = run_evaluation(
            processed_data_dir=str(data_path),
            model_path=str(model_path),
            feature_sizes=feature_sizes,
            k_list=[5, 10],
            use_cuda=use_cuda,
        )
        # model_baseline: exp.md 기준 참조값 (하드코딩)
        run_metrics["model_baseline"] = {
            "auc": 0.6426139601139602,
            "ndcg@5": 0.22093023283013208,
            "ndcg@10": 0.22093023283013208,
            "recall@5": 0.22093023283013208,
            "recall@10": 0.22093023283013208,
        }
        popularity_baseline = run_popularity_baseline(
            processed_data_dir=str(data_path),
            k_list=[5, 10],
        )
        if "error" not in popularity_baseline:
            run_metrics["popularity_baseline"] = popularity_baseline
            print(f"Popularity baseline: ndcg@5={popularity_baseline.get('ndcg@5', 0):.4f}, recall@5={popularity_baseline.get('recall@5', 0):.4f}")
        else:
            run_metrics["popularity_baseline_error"] = popularity_baseline.get("error", "unknown")
        random_baseline = run_random_baseline(
            processed_data_dir=str(data_path),
            k_list=[5, 10],
            seed=42,
        )
        if "error" not in random_baseline:
            run_metrics["random_baseline"] = random_baseline
            print(f"Random baseline: ndcg@5={random_baseline.get('ndcg@5', 0):.4f}, recall@5={random_baseline.get('recall@5', 0):.4f}")
        else:
            run_metrics["random_baseline_error"] = random_baseline.get("error", "unknown")
        if "error" not in run_metrics:
            metrics_path = out_run / "run_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(run_metrics, f, ensure_ascii=False, indent=2)
            print(f"Offline metrics saved: {metrics_path}")

    run_manifest = {
        "pipeline_version": pipeline_version,
        "model_path": str(model_path),
        "feature_sizes_path": str(meta_path),
        "processed_data_dir": processed_data_dir,
        "metrics": run_metrics,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_run / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)
    print(f"Run manifest: {manifest_path}")

    if use_wandb and wandb_logger.is_available():
        wandb_logger.log_artifact(
            "model_checkpoint",
            "model",
            out_run,
            metadata={"pipeline_version": pipeline_version, **train_config},
        )
        if run_metrics and "error" not in run_metrics:
            wandb_logger.log_metrics(run_metrics)
            wandb_logger.log_artifact(
                "evaluation_report",
                "evaluation",
                out_run / "run_metrics.json",
                metadata={"pipeline_version": pipeline_version},
            )
        wandb_logger.set_summary("pipeline_version", pipeline_version)
        wid = wandb_logger.get_run_id()
        if wid:
            wandb_logger.set_summary("wandb_run_id", wid)
            run_manifest["wandb_run_id"] = wid
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    return {
        "model_path": str(model_path),
        "feature_sizes_path": str(meta_path),
        "processed_data_dir": processed_data_dir,
        "pipeline_version": pipeline_version,
        "run_manifest_path": str(manifest_path),
        "metrics": run_metrics,
    }


def _default_source_dataset_path() -> str:
    return str(_deepfm_root / "data" / "training_dataset.csv")


@flow(name="DeepFM Training Pipeline", log_prints=True)
def deepfm_training_flow(
    raw_data_dir: str | None = None,
    processed_data_dir: str | None = None,
    source_dataset_path: str | None = None,
    test_ratio: float = 0.2,
    random_state: int | None = 42,
    num_train_sample: int | None = 9000,
    num_test_sample: int | None = 1000,
    num_val: int = 1000,
    epochs: int = 5,
    batch_size: int = 100,
    lr: float = 1e-4,
    output_dir: str | None = None,
    use_cuda: bool = False,
    skip_preprocess: bool = False,
    use_sample_weight: bool = True,
    time_column: str | None = None,
    train_end: str | None = None,
    valid_end: str | None = None,
    test_end: str | None = None,
    group_column: str | None = None,
    negative_sampling_ratio: float = 1.0,
    negative_sampling_seed: int = 42,
    eval_list_size: int = 101,
    eval_num_neg: int = 100,
    eval_num_popular_neg: int = 50,
    eval_popular_top_k: int = 1000,
    eval_list_seed: int = 42,
    use_wandb: bool = True,
    seed: int = 42,
    exp_ablation: list[str] | None = None,
) -> dict:
    """
    (선택) train/test 분할 → 전처리 → 학습을 순서대로 실행하는 DeepFM 파이프라인.
    source_dataset_path를 주면 해당 CSV를 train.csv/test.csv로 나눈 뒤 전처리·학습.
    (wandb_design) use_wandb=True 시 Artifacts·메트릭 로깅, pipeline_version 매핑.
    """
    from utils import wandb_logger

    raw = raw_data_dir or _default_raw_data_dir()
    processed = processed_data_dir or _default_processed_data_dir()
    out = output_dir or _default_output_dir()

    if use_wandb and wandb_logger.is_available():
        run_ctx = get_run_context()
        run_id = getattr(run_ctx, "flow_run", None) and getattr(run_ctx.flow_run, "id", None)
        run_name = str(run_id) if run_id else "manual"
        wandb_logger.init_run(
            project="deepfm-pipeline",
            config={
                "raw_data_dir": raw,
                "processed_data_dir": processed,
                "source_dataset_path": source_dataset_path,
                "test_ratio": test_ratio,
                "num_train_sample": num_train_sample,
                "num_test_sample": num_test_sample,
                "num_val": num_val,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "time_column": time_column,
                "train_end": train_end,
                "valid_end": valid_end,
                "test_end": test_end,
                "group_column": group_column,
                "negative_sampling_ratio": negative_sampling_ratio,
                "eval_list_size": eval_list_size,
                "eval_num_neg": eval_num_neg,
                "eval_num_popular_neg": eval_num_popular_neg,
                "eval_popular_top_k": eval_popular_top_k,
                "seed": seed,
            },
            run_name=run_name,
        )

    if not skip_preprocess:
        if source_dataset_path is not None:
            split_train_test_task(
                source_dataset_path=source_dataset_path,
                raw_data_dir=raw,
                test_ratio=test_ratio,
                random_state=random_state,
                time_column=time_column,
            )
        preprocess_task(
            raw_data_dir=raw,
            processed_data_dir=processed,
            num_train_sample=num_train_sample,
            num_test_sample=num_test_sample,
            use_sample_weight=use_sample_weight,
            time_column=time_column,
            train_end=train_end,
            valid_end=valid_end,
            test_end=test_end,
            group_column=group_column,
            negative_sampling_ratio=negative_sampling_ratio,
            negative_sampling_seed=negative_sampling_seed,
            eval_list_size=eval_list_size,
            eval_num_neg=eval_num_neg,
            eval_num_popular_neg=eval_num_popular_neg,
            eval_popular_top_k=eval_popular_top_k,
            eval_list_seed=eval_list_seed,
            use_wandb=use_wandb,
            exp_ablation=exp_ablation,
        )

    result = train_task(
        processed_data_dir=processed,
        num_train=num_train_sample,
        num_val=num_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=out,
        use_cuda=use_cuda,
        verbose=True,
        use_wandb=use_wandb,
        seed=seed,
    )
    if use_wandb and wandb_logger.is_available():
        wandb_logger.finish_run()
    return result


@task(name="deepfm-score-batch", log_prints=True)
def score_batch_task(
    run_dir: str,
    candidates_path: str,
    output_path: str,
    meta_path: str | None = None,
    ttl_hours: float = 24.0,
    batch_size: int = 256,
    use_wandb: bool = True,
) -> dict:
    """
    배치 스코어링 실행 후 recommendation 형식 CSV 출력.
    (wandb_design) scoring_output artifact + pipeline_version 매핑.
    """
    from utils.score_batch import run as score_batch_run
    from utils import wandb_logger

    score_batch_run(
        run_dir=Path(run_dir),
        candidates_path=Path(candidates_path),
        output_path=Path(output_path),
        meta_path=Path(meta_path) if meta_path else None,
        ttl_hours=ttl_hours,
        batch_size=batch_size,
    )
    pipeline_version = "unknown"
    pv_path = Path(run_dir) / "pipeline_version.txt"
    if pv_path.exists():
        pipeline_version = pv_path.read_text(encoding="utf-8").strip()

    if use_wandb and wandb_logger.is_available():
        wandb_logger.log_artifact(
            "scoring_output",
            "scoring",
            output_path,
            metadata={"pipeline_version": pipeline_version},
        )
    return {"output_path": output_path, "pipeline_version": pipeline_version}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="DeepFM 학습 파이프라인 (분할 → 전처리 → 학습)")
    p.add_argument("--source", type=str, default=None, help="단일 CSV 경로. 없으면 data/training_dataset.csv 사용 시도")
    p.add_argument("--raw-dir", type=str, default=None, help="raw 데이터 디렉터리 (train.csv, test.csv)")
    p.add_argument("--processed-dir", type=str, default=None, help="전처리 결과 출력 디렉터리")
    p.add_argument("--output-dir", type=str, default=None, help="모델/아티팩트 출력 디렉터리")
    p.add_argument("--test-ratio", type=float, default=0.2, help="source 사용 시 test 비율 (0~1)")
    p.add_argument("--random-state", type=int, default=42, help="train/test 분할 시드")
    p.add_argument("--num-train", type=int, default=None, help="학습 샘플 수 제한")
    p.add_argument("--num-test", type=int, default=None, help="테스트 샘플 수 제한")
    p.add_argument("--num-val", type=int, default=1000, help="validation 샘플 수")
    p.add_argument("--epochs", type=int, default=5, help="학습 epoch 수")
    p.add_argument("--batch-size", type=int, default=100, help="배치 크기")
    p.add_argument("--lr", type=float, default=1e-4, help="학습률")
    p.add_argument("--negative-ratio", type=float, default=1.0, help="positive 1건당 음성 샘플 수 (0이면 미적용)")
    p.add_argument("--negative-seed", type=int, default=42, help="음성 샘플링 시드")
    p.add_argument("--eval-list-size", type=int, default=101, help="test/val 리스트당 행 수 (1 pos + eval-num-neg neg). 0이면 미적용")
    p.add_argument("--eval-num-neg", type=int, default=100, help="리스트당 음성 개수 (eval-list-size-1)")
    p.add_argument("--eval-num-popular-neg", type=int, default=50, help="리스트당 인기 아이템 기반 음성 개수 (나머지는 랜덤)")
    p.add_argument("--eval-popular-top-k", type=int, default=1000, help="인기 아이템 풀 크기 (positive count 상위 K)")
    p.add_argument("--eval-list-seed", type=int, default=42, help="eval 리스트 구성 시드")
    p.add_argument("--exp-ablation", type=str, default=None, help="exp 피처 ablation (쉼표 구분, 예: user_category_match,user_region_match,price_diff)")
    p.add_argument("--seed", type=int, default=42, help="학습 재현성 시드 (모델 초기화·배치 셔플)")
    p.add_argument("--skip-preprocess", action="store_true", help="전처리 생략 (기존 train.txt 사용)")
    p.add_argument("--no-wandb", action="store_true", help="wandb 로깅 비활성화")
    p.add_argument("--cuda", action="store_true", help="CUDA 사용")
    args = p.parse_args()

    default_source = _default_source_dataset_path()
    source = args.source if args.source is not None else (default_source if Path(default_source).exists() else None)
    deepfm_training_flow(
        source_dataset_path=source,
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        num_train_sample=args.num_train,
        num_test_sample=args.num_test,
        num_val=args.num_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        negative_sampling_ratio=args.negative_ratio,
        negative_sampling_seed=args.negative_seed,
        eval_list_size=args.eval_list_size,
        eval_num_neg=args.eval_num_neg,
        eval_num_popular_neg=args.eval_num_popular_neg,
        eval_popular_top_k=args.eval_popular_top_k,
        eval_list_seed=args.eval_list_seed,
        seed=args.seed,
        exp_ablation=[s.strip() for s in args.exp_ablation.split(",")] if args.exp_ablation else None,
        skip_preprocess=args.skip_preprocess,
        use_wandb=not args.no_wandb,
        use_cuda=args.cuda,
    )
