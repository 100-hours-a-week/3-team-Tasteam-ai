TASK

1. 감성 분석
- 배치 단위로 llm에게 전달
- llm이 배치 단위로 긍/부정 개수 세고, 총 긍/부정 개수 반환.
2. 요약
- 긍/부정 리뷰 각각에 대해 벡터 검색을 수행해 긍정 3개, 부정 3개 반환.
- llm이 그 긍/부정 리뷰들을 보고 총 요약 수행.

3. 강점 추출
- vectordb가 강점 추출 대상 음식점의 긍정 리뷰 추출 target_review
- vectordb가 비교 대상 음식점들 각각에 벡터 검색을 수행해 긍정 리뷰 추출 com_reivew
- llm이 이 target_review,com_review를 보고 target_review의 강점 추출

4. 이미지 검색
