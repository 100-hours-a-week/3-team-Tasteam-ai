#!/usr/bin/env python3
"""
감성 1차 분류 점수 분포 및 LLM 재판정 대상 개수 확인용 테스트 스크립트.

현재 로직: positive_score > 0.8 → positive(재판정 없음), 그 외 전부 → LLM 재판정 대상.
이 스크립트로 1차 모델 점수 분포를 보고, 필요 시 threshold(예: 0.3) 추가 검토에 활용.

사용 예:
  # 파일에서 리뷰 텍스트 읽기 (한 줄당 한 리뷰)
  python scripts/test_sentiment_threshold.py --file path/to/reviews.txt

  # 인라인 샘플
  python scripts/test_sentiment_threshold.py --samples "맛있어요" "별로에요" "그냥 그래요"

  # 특정 레스토랑 (로컬 Qdrant + 벡터 DB)
  python scripts/test_sentiment_threshold.py --restaurant-id 4 --local

  # test_data_sample.json에서 restaurant_id=4만 사용 (기본 동작)
  python scripts/test_sentiment_threshold.py --from-test-data 4
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 샘플만 쓸 때는 transformers 등 로딩 전에 --file/--samples 여부만 확인 가능
DEFAULT_THRESHOLD = 0.8
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "data" / "test_data_sample.json"


def load_content_from_file(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_content_from_test_data(data_path: Path, restaurant_id: int) -> list[str]:
    """test_data_sample.json 등 JSON에서 해당 restaurant_id 리뷰만 추출해 content 리스트 반환."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    contents: list[str] = []
    if isinstance(data, list):
        for r in data:
            rid = r.get("restaurant_id") if isinstance(r, dict) else getattr(r, "restaurant_id", None)
            if rid != restaurant_id:
                continue
            c = r.get("content", "") if isinstance(r, dict) else getattr(r, "content", "")
            if c:
                contents.append(c)
    elif isinstance(data, dict) and "restaurants" in data:
        for rest in data["restaurants"] or []:
            rid = rest.get("id") or rest.get("restaurant_id")
            if rid != restaurant_id:
                continue
            for r in rest.get("reviews") or []:
                c = r.get("content", "") if isinstance(r, dict) else getattr(r, "content", "")
                if c:
                    contents.append(c)
            break
    return contents


def fetch_reviews_from_local_db(restaurant_id: int) -> list[str]:
    """로컬 Qdrant + VectorSearch로 해당 레스토랑 리뷰 조회 후 content 리스트 반환."""
    from src.api.dependencies import get_qdrant_client
    from src.config import Config
    from src.vector_search import VectorSearch
    from src.review_utils import extract_content_list

    client = get_qdrant_client()
    vs = VectorSearch(
        qdrant_client=client,
        collection_name=getattr(Config, "COLLECTION_NAME", "reviews_collection"),
    )
    reviews = vs.get_restaurant_reviews(str(restaurant_id))
    return extract_content_list(reviews)


def run_hf_scores(content_list: list[str], threshold: float, batch_size: int):
    from src.sentiment_analysis import SentimentAnalyzer

    if not content_list:
        print("리뷰가 없습니다.", file=sys.stderr)
        return
    analyzer = SentimentAnalyzer(vector_search=None, llm_utils=None)
    pipe = analyzer._get_sentiment_pipeline()
    positive_threshold = threshold
    results = []  # (idx, positive_score, label, for_llm)

    for i in range(0, len(content_list), batch_size):
        batch = content_list[i : i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(content_list))))
        outputs = pipe(batch, top_k=None, truncation=True, max_length=512)
        for idx, out in zip(batch_indices, outputs):
            if isinstance(out, list) and len(out) >= 2:
                # 레이블 문자열로 긍정 점수 조회 (모델 출력 순서는 점수/라벨에 따라 달라질 수 있음)
                raw = 0.0
                for item in out:
                    if not isinstance(item, dict):
                        continue
                    lab = item.get("label", "")
                    if analyzer._map_label_to_binary(lab) == "positive":
                        raw = item.get("score", 0.0)
                        break
                if raw == 0.0 and len(out) >= 2:
                    raw = out[1].get("score", 0.0)  # fallback
                positive_score = analyzer._score_to_float(raw)
                is_positive = positive_score > positive_threshold
                label = "positive" if is_positive else "negative"
                for_llm = not is_positive
                results.append((idx, positive_score, label, for_llm))
            else:
                results.append((idx, 0.0, "unknown", True))

    n_total = len(content_list)
    n_positive = sum(1 for _, _, lab, for_llm in results if not for_llm)
    n_for_llm = sum(1 for _, _, _, for_llm in results if for_llm)

    print(f"\n=== 1차 분류 요약 (threshold={positive_threshold}) ===")
    print(f"총 리뷰: {n_total}")
    print(f"positive(재판정 제외): {n_positive}")
    print(f"LLM 재판정 대상: {n_for_llm}")

    # 점수 구간별 개수 (선택적: 0.3 미만 확정 negative 가정 시)
    bands = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
    print("\n--- positive_score 구간별 개수 ---")
    for lo, hi in bands:
        c = sum(1 for _, s, _, _ in results if lo <= s < hi)
        label_band = "확정 negative(재판정 불필요 가정)" if hi <= 0.3 else "LLM 재판정 대상" if hi <= 0.8 else "positive"
        print(f"  [{lo:.1f}, {hi:.1f}): {c}개  ({label_band})")

    print("\n--- 리뷰별 점수 (앞 20개 + 마지막 5개) ---")
    for idx, score, label, for_llm in results[:20]:
        llm_mark = " [LLM]" if for_llm else ""
        preview = (content_list[idx][:40] + "…") if len(content_list[idx]) > 40 else content_list[idx]
        print(f"  {idx}: {score:.3f} {label}{llm_mark}  {preview}")
    if len(results) > 25:
        print("  ...")
        for idx, score, label, for_llm in results[-5:]:
            llm_mark = " [LLM]" if for_llm else ""
            preview = (content_list[idx][:40] + "…") if len(content_list[idx]) > 40 else content_list[idx]
            print(f"  {idx}: {score:.3f} {label}{llm_mark}  {preview}")

    # 제안: 0.3 미만이면 재판정 제외 시
    n_ambiguous = sum(1 for _, s, _, for_llm in results if for_llm and 0.3 <= s < 0.8)
    n_definite_neg = sum(1 for _, s, _, for_llm in results if for_llm and s < 0.3)
    print(f"\n--- 참고: 0.3 미만을 확정 negative로 두면 ---")
    print(f"  확정 negative(재판정 제외): {n_definite_neg}개")
    print(f"  애매 구간만 LLM 재판정: {n_ambiguous}개")


def main():
    parser = argparse.ArgumentParser(
        description="감성 1차 분류 점수 분포 및 LLM 재판정 대상 개수 확인"
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", "-f", type=Path, help="리뷰 텍스트 파일 (한 줄당 한 리뷰)")
    g.add_argument("--samples", "-s", nargs="+", help="인라인 리뷰 텍스트 목록")
    g.add_argument("--restaurant-id", "-r", type=int, help="레스토랑 ID (--local 시 로컬 Qdrant에서 조회)")
    g.add_argument("--from-test-data", type=int, metavar="RESTAURANT_ID", help="data/test_data_sample.json에서 해당 레스토랑 ID만 사용 (예: 4)")
    parser.add_argument("--data-file", type=Path, default=None, help="--from-test-data 사용 시 JSON 경로 (기본: data/test_data_sample.json)")
    parser.add_argument("--local", "-l", action="store_true", help="--restaurant-id와 함께 사용 시 로컬 Qdrant/벡터 DB에서 리뷰 로드")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="positive 판정 기준 (기본 0.8)")
    parser.add_argument("--batch-size", type=int, default=10, help="HF 배치 크기")
    args = parser.parse_args()

    content_list: list[str] = []
    if args.file:
        content_list = load_content_from_file(args.file)
        print(f"파일에서 {len(content_list)}개 리뷰 로드: {args.file}")
    elif args.samples:
        content_list = [s.strip() for s in args.samples if s.strip()]
        print(f"샘플 {len(content_list)}개 사용")
    elif args.from_test_data is not None:
        data_path = args.data_file or DEFAULT_TEST_DATA_PATH
        if not data_path.is_file():
            parser.error(f"데이터 파일이 없습니다: {data_path}")
        content_list = load_content_from_test_data(data_path, args.from_test_data)
        print(f"test_data_sample에서 restaurant_id={args.from_test_data} 리뷰 {len(content_list)}개 로드 ({data_path})")
    elif args.restaurant_id is not None:
        if not args.local:
            parser.error("--restaurant-id 사용 시 로컬 DB에서 조회하려면 --local 을 지정하세요.")
        content_list = fetch_reviews_from_local_db(args.restaurant_id)
        print(f"로컬 DB에서 restaurant_id={args.restaurant_id} 리뷰 {len(content_list)}개 로드")
    else:
        parser.error("--file, --samples, --from-test-data, --restaurant-id 중 하나 필요")

    if not content_list:
        print("리뷰가 없습니다.", file=sys.stderr)
        sys.exit(1)
    run_hf_scores(content_list, args.threshold, args.batch_size)


if __name__ == "__main__":
    main()
