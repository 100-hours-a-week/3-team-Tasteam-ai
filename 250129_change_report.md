1. API 입출력 형식 변경

- (기존) 모든 메타 데이터 포함 -> (현재) 실제 API 작동에 필요한 데이터와 최소한의 메타데이터만 남김(ID)

2. 미사용 레거시 코드 제거, 문서 언급 삭제
- Strength API
-- min_support
-- strength_type(representative / distinct / both)

- Vector API
-- expand_query

3. 기능 삭제
- image_search 기능 삭제 (코드,문서 언급 모두 삭제)
-- 기능에 대한 기획,설계가 구체적으로 되지 않았음.