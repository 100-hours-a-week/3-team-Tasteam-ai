"""
배치 추천 생성 파이프라인 (deepfm_design.md §6-3).

사용자·아이템·컨텍스트 후보에 대해 DeepFM score 예측 후 TopN 정렬,
score / rank / context_snapshot / pipeline_version / generated_at / expires_at 출력.
→ recommendation 테이블 insert는 호출 측(ETL/DB)에서 수행.

사용 예:
  python -m utils.score_batch --run-dir output/manual --candidates data/candidates.txt --meta data/candidates_meta.csv --out output/recommendations.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# deepfm_training 루트
_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dataset import _get_continuous_feature_count
from model.DeepFM import DeepFM
from utils.dataPreprocess import raw_rows_to_feature_matrix
from utils.raw_to_pipeline import read_table


def load_run(run_dir: Path) -> tuple[DeepFM, list[int], str]:
    """run_dir에서 model.pt, feature_sizes.txt, pipeline_version.txt 로드."""
    model_path = run_dir / "model.pt"
    sizes_path = run_dir / "feature_sizes.txt"
    version_path = run_dir / "pipeline_version.txt"
    if not model_path.exists() or not sizes_path.exists():
        raise FileNotFoundError(f"Run dir must contain model.pt and feature_sizes.txt: {run_dir}")
    feature_sizes = [int(x) for x in np.loadtxt(str(sizes_path), delimiter=",").ravel()]
    pipeline_version = version_path.read_text(encoding="utf-8").strip() if version_path.exists() else "deepfm-1.0.unknown"
    model = DeepFM(feature_sizes, use_cuda=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, feature_sizes, pipeline_version


def score_candidates(
    model: DeepFM,
    feature_sizes: list[int],
    candidates: np.ndarray,
    n_cont: int,
    batch_size: int = 256,
) -> np.ndarray:
    """후보 행렬(연속+범주 인덱스)에 대해 예측 점수 반환."""
    preds = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            Xi_cont = np.zeros((len(batch), n_cont))
            Xi_cat = batch[:, n_cont:]
            Xi = torch.from_numpy(
                np.concatenate([Xi_cont, Xi_cat], axis=1).astype(np.int32)
            ).unsqueeze(-1)
            Xv_cont = batch[:, :n_cont].astype(np.float32)
            Xv_cat = np.ones_like(Xi_cat, dtype=np.float32)
            Xv = torch.from_numpy(np.concatenate([Xv_cont, Xv_cat], axis=1))
            out = model(Xi, Xv)
            preds.append(torch.sigmoid(out).cpu().numpy().ravel())
    return np.concatenate(preds)


def run(
    run_dir: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    meta_path: str | Path | None = None,
    raw_candidates_path: str | Path | None = None,
    ttl_hours: float = 24.0,
    batch_size: int = 256,
    output_format: str = "csv",
) -> None:
    """
    후보에 대해 DeepFM 점수 예측 → (user/anon, restaurant)별 rank 부여 → recommendation 형식 출력.

    - candidates_path: CSV, 컬럼 수 = feature_sizes 개수 (연속+범주 인덱스만, 헤더 없음 또는 있음).
      raw_candidates_path가 있으면 무시됨.
    - raw_candidates_path: (선택) 전처리 전 raw CSV 경로. run_dir의 categorical_dicts.json으로 인코딩 후 추론.
      지정 시 meta는 raw CSV의 member_id/user_id, anonymous_id, restaurant_id에서 추출.
    - meta_path: CSV, 컬럼 user_id, anonymous_id, restaurant_id, context_snapshot (선택). raw_candidates_path 미사용 시만 사용.
    - output_path: recommendation 행 저장 경로 (.csv 또는 .json.gz)
    - output_format: "csv" | "json.gz" — 출력 파일 형식.
    """
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    model, feature_sizes, pipeline_version = load_run(run_dir)
    n_cont = _get_continuous_feature_count(str(run_dir))
    n_fields = len(feature_sizes)

    if raw_candidates_path is not None:
        raw_path = Path(raw_candidates_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"raw_candidates_path not found: {raw_path}")
        raw_df = read_table(raw_path)
        raw_rows = raw_df.to_dict("records")
        feature_rows = raw_rows_to_feature_matrix(raw_rows, run_dir)
        X = np.array(feature_rows, dtype=np.float32)
        if X.shape[1] != n_fields:
            raise ValueError(f"Raw→feature columns {X.shape[1]}, expected {n_fields} (feature_sizes)")
        meta_from_raw = True
    else:
        candidates_path = Path(candidates_path)
        df = read_table(candidates_path)
        if df.shape[1] != n_fields:
            raise ValueError(f"Candidates have {df.shape[1]} columns, expected {n_fields} (feature_sizes)")
        X = df.values.astype(np.float32)
        meta_from_raw = False

    scores = score_candidates(model, feature_sizes, X, n_cont, batch_size=batch_size)

    generated_at = datetime.now(timezone.utc)
    expires_at = generated_at + timedelta(hours=ttl_hours)
    gen_ts = generated_at.isoformat()
    exp_ts = expires_at.isoformat()

    if meta_from_raw:
        user_id = raw_df.get("user_id", raw_df.get("member_id", pd.Series([""] * len(raw_df))))
        anonymous_id = raw_df.get("anonymous_id", pd.Series([""] * len(raw_df)))
        restaurant_id = raw_df["restaurant_id"] if "restaurant_id" in raw_df.columns else list(range(len(scores)))
        context_snapshot = raw_df.get("context_snapshot", pd.Series(["{}"] * len(raw_df)))
    elif meta_path and Path(meta_path).exists():
        meta = read_table(Path(meta_path))
        if len(meta) != len(scores):
            raise ValueError(f"Meta rows {len(meta)} != candidate rows {len(scores)}")
        user_id = meta.get("user_id", meta.get("member_id", pd.Series([""] * len(meta))))
        anonymous_id = meta.get("anonymous_id", pd.Series([""] * len(meta)))
        restaurant_id = meta["restaurant_id"] if "restaurant_id" in meta.columns else list(range(len(scores)))
        context_snapshot = meta.get("context_snapshot", pd.Series(["{}"] * len(meta)))
    else:
        user_id = [""] * len(scores)
        anonymous_id = [""] * len(scores)
        restaurant_id = list(range(len(scores)))
        context_snapshot = ["{}"] * len(scores)

    key = []
    for i in range(len(scores)):
        uid = str(user_id.iloc[i]) if hasattr(user_id, "iloc") else str(user_id[i])
        aid = str(anonymous_id.iloc[i]) if hasattr(anonymous_id, "iloc") else str(anonymous_id[i])
        key.append(uid if (uid and uid != "nan") else f"a_{aid}")

    df_out = pd.DataFrame({
        "user_id": user_id,
        "anonymous_id": anonymous_id,
        "restaurant_id": restaurant_id,
        "score": scores,
        "context_snapshot": context_snapshot,
    })
    df_out["_user_key"] = key
    df_out["rank"] = 0
    df_out["pipeline_version"] = pipeline_version
    df_out["generated_at"] = gen_ts
    df_out["expires_at"] = exp_ts

    for _, grp in df_out.groupby("_user_key", sort=False):
        order = grp["score"].values.argsort()[::-1]
        idx = grp.index[order]
        for r, i in enumerate(idx, start=1):
            df_out.loc[i, "rank"] = r

    df_out = df_out.sort_values(["_user_key", "rank"]).drop(columns=["_user_key"])

    # restaurant_id가 있는 행만 내보냄 (유효한 추천만)
    def _not_empty(v) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        s = str(v).strip()
        return s != "" and s.lower() != "nan"
    mask = df_out["restaurant_id"].apply(_not_empty)
    df_out = df_out.loc[mask].copy()

    # user_id를 정수로 캐스팅 (숫자면 int, 아니면 빈 문자열 유지)
    u = pd.to_numeric(df_out["user_id"], errors="coerce")
    df_out["user_id"] = u.apply(lambda x: int(x) if pd.notna(x) else "")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format.lower() == "json.gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as zf:
            for _, row in df_out.iterrows():
                d = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()}
                zf.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        df_out.to_csv(output_path, index=False)
    print(f"Wrote {len(df_out)} recommendation rows to {output_path}, pipeline_version={pipeline_version}")


def main() -> None:
    p = argparse.ArgumentParser(description="DeepFM batch scoring → recommendation CSV (§6-3)")
    p.add_argument("--run-dir", type=str, default=None, help="Run directory (model.pt, feature_sizes.txt 등). --pipeline-version과 둘 중 하나 필수.")
    p.add_argument("--pipeline-version", type=str, default=None, help="run_dir 없을 때 로컬 output/ 또는 wandb artifact에서 해당 버전 사용")
    p.add_argument("--candidates", type=str, default=None, help="Candidates CSV (preprocessed feature columns). --raw-candidates 사용 시 생략 가능.")
    p.add_argument("--raw-candidates", type=str, default=None, help="Raw CSV 경로. run_dir의 vocab으로 인코딩 후 추론. meta는 raw CSV 컬럼에서 추출.")
    p.add_argument("--meta", type=str, default=None, help="Optional meta CSV: user_id, anonymous_id, restaurant_id (--raw-candidates 미사용 시)")
    p.add_argument("--out", type=str, required=True, help="Output path (.csv or .json.gz)")
    p.add_argument("--output-format", choices=["csv", "json.gz"], default="csv", help="Output file format")
    p.add_argument("--ttl-hours", type=float, default=24.0, help="TTL hours for expires_at")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--artifact-cache-dir", type=str, default=None, help="아티팩트 다운로드 캐시 (pipeline_version 사용 시)")
    p.add_argument("--wandb-project", type=str, default=None, help="W&B 프로젝트 (아티팩트 조회 시)")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B 엔티티 (아티팩트 조회 시)")
    args = p.parse_args()
    if not args.candidates and not args.raw_candidates:
        p.error("One of --candidates or --raw-candidates is required")
    if not args.run_dir and not args.pipeline_version:
        p.error("One of --run-dir or --pipeline-version is required")

    run_dir_arg = Path(args.run_dir) if args.run_dir else None
    search_dir = _ROOT / "output"
    cache_dir = Path(args.artifact_cache_dir) if args.artifact_cache_dir else (search_dir / "artifact_cache")
    from utils.run_dir_resolver import resolve_run_dir
    try:
        resolved_run_dir = resolve_run_dir(
            run_dir=run_dir_arg,
            pipeline_version=args.pipeline_version,
            search_output_dir=search_dir,
            cache_dir=cache_dir,
            wandb_project=args.wandb_project or "deepfm-pipeline",
            wandb_entity=args.wandb_entity,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    run(
        run_dir=resolved_run_dir,
        candidates_path=args.candidates or "",
        output_path=args.out,
        meta_path=args.meta,
        raw_candidates_path=args.raw_candidates,
        ttl_hours=args.ttl_hours,
        batch_size=args.batch_size,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
