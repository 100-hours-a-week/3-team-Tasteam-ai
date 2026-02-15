"""
최신 파이프라인만 사용: strength_in_aspect (Kiwi+lift, recall_seeds, strength_display) + optional final_summary.

- run_strength_in_aspect: 전체 vs 타겟 lift, recall_seeds_for_summary, strength_display
- recall_seeds_for_summary → final_summary_pipeline.main() 하이브리드 쿼리로 전달 (선택)
"""

import time
import os
import sys
from pathlib import Path

# final_pipeline 내부에서 실행 시 동일 디렉터리
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from strength_in_aspect import run_strength_in_aspect

# 경로 (프로젝트 루트 기준)
_project = _here.parents[1]
aspect_data = _project / "data" / "kr3.tsv"
if not aspect_data.exists():
    aspect_data = _project / "data" / "test_data_sample.json"
stopwords_data = _project / "hybrid_search" / "data_preprocessing" / "stopwords-ko.txt"
target_data = _project / "data" / "test_data_sample.json"
target_rid = 4

if __name__ == "__main__":
    t0 = time.perf_counter()

    res = run_strength_in_aspect(
        str(aspect_data),
        str(stopwords_data),
        str(target_data),
        target_rid=target_rid,
    )

    print("lift_service:", res["lift_service"])
    print("lift_price:", res["lift_price"])
    print("strength_display:")
    for s in res.get("strength_display", []):
        print(" ", s)
    print("recall_seeds_for_summary keys:", list((res.get("recall_seeds_for_summary") or {}).keys()))

    # recall_seeds → final_summary 하이브리드 쿼리로 전달 (선택)
    try:
        from final_summary_pipeline import main as summary_main
        summary_main(recall_seeds_for_summary=res["recall_seeds_for_summary"])
        print("recall_seeds → final_summary_pipeline.main() 전달 완료")
    except Exception as e:
        print("final_summary_pipeline.main 생략:", e)

    print("elapsed:", round(time.perf_counter() - t0, 4), "s")
