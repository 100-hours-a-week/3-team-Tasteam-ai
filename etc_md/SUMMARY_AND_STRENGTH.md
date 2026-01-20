할루시네이션 방지 방법이 다름. 그 이유
260116

SUMMARY

1. 대표 벡터 top_k
2. 여기서 llm이 요약 수행
3. 여기서 llm이 aspect 추출
4. aspect의 claim 대표 벡터 top_k와 cos 유사도로 할루시네이션 여부 확인
5. 할루시네이션 아닌건 유지,
   할루시네이션 시 llm이 더 일반적인 문장으로 바꾸기.
6. 이 claim들 반환.

Stength
전체 리뷰에서 근거 문장을 수집하는 이유
할루시네이션 방지: support_count < min_support이면 버림
대표 벡터 생성: evidence_reviews의 centroid로 대표 벡터 생성 → Step D, Step G에서 사용
정량적 대표성: support_count로 대표성 측정 및 최종 점수 계산에 사용
일관성 검증: consistency 계산으로 근거 리뷰들의 일관성 확인
최근성 반영: recency 계산으로 최근 리뷰에 높은 가중치
Evidence Overlap: Step D에서 과병합 방지 가드레일로 사용
