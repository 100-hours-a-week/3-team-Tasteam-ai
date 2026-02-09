-- AI API 결과 저장용 DDL (PostgreSQL 17)
-- 전제: restaurant 테이블이 이미 존재하며, restaurant_ai_results.restaurant_id는 restaurant.id를 참조한다.
-- tasteam_app_all_restaurants_ai_api_results.json 기반 DML은 scripts/json_results_to_dml.py로 생성 가능.

-- ---------------------------------------------------------------------------
-- run_meta: 배치 실행 메타정보 (API base_url, 요청 시각, 배치 크기 등)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_meta (
  id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  base_url        VARCHAR(512),
  total_restaurants INTEGER,
  requested_at    TIMESTAMP WITH TIME ZONE,
  batch_size      INTEGER,
  apis_json       TEXT,
  created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE run_meta IS 'AI API 배치 실행 메타정보';
COMMENT ON COLUMN run_meta.apis_json IS '호출 API 목록 JSON 배열, 예: ["summary","sentiment","comparison"]';

-- ---------------------------------------------------------------------------
-- restaurant_ai_results: 음식점별 AI API 결과 (요약/감성/비교 JSON)
-- FK: restaurant_id → restaurant.id (restaurant 테이블 선행 필요)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS restaurant_ai_results (
  id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  restaurant_id   BIGINT NOT NULL REFERENCES restaurant(id),
  restaurant_name VARCHAR(100),
  summary_json    TEXT,
  sentiment_json  TEXT,
  comparison_json TEXT,
  errors_json     TEXT,
  created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE restaurant_ai_results IS '음식점별 AI API 결과 (요약/감성/비교)';
COMMENT ON COLUMN restaurant_ai_results.restaurant_id IS 'FK → restaurant.id';
COMMENT ON COLUMN restaurant_ai_results.summary_json IS '요약 API 응답 JSON';
COMMENT ON COLUMN restaurant_ai_results.sentiment_json IS '감성 분석 API 응답 JSON';
COMMENT ON COLUMN restaurant_ai_results.comparison_json IS '비교 API 응답 JSON';
COMMENT ON COLUMN restaurant_ai_results.errors_json IS '에러 정보 JSON (없으면 빈 문자열 또는 NULL)';

-- 인덱스: 음식점 기준 조회
CREATE INDEX IF NOT EXISTS idx_restaurant_ai_results_restaurant_id
  ON restaurant_ai_results(restaurant_id);

CREATE INDEX IF NOT EXISTS idx_restaurant_ai_results_created_at
  ON restaurant_ai_results(created_at DESC);
