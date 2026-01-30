| idx | model                                                       | license      | size_in_GB | dim  | description (요약)                     |
| --: | ----------------------------------------------------------- | ------------ | ---------- | ---- | ------------------------------------ |
|   0 | BAAI/bge-small-en-v1.5                                      | mit          | 0.067      | 384  | English text embeddings              |
|   1 | sentence-transformers/all-MiniLM-L6-v2                      | apache-2.0   | 0.090      | 384  | English sentence embeddings          |
|   2 | BAAI/bge-small-zh-v1.5                                      | mit          | 0.090      | 512  | Chinese text embeddings              |
|   3 | snowflake/snowflake-arctic-embed-xs                         | apache-2.0   | 0.090      | 384  | English text embeddings              |
|   4 | jinaai/jina-embeddings-v2-small-en                          | apache-2.0   | 0.120      | 512  | English text embeddings              |
|   5 | BAAI/bge-small-en                                           | mit          | 0.130      | 384  | English text embeddings              |
|   6 | nomic-ai/nomic-embed-text-v1.5-Q                            | apache-2.0   | 0.130      | 768  | Multimodal (text/image), English     |
|   7 | snowflake/snowflake-arctic-embed-s                          | apache-2.0   | 0.130      | 384  | English text embeddings              |
|   8 | BAAI/bge-base-en-v1.5                                       | mit          | 0.210      | 768  | English text embeddings              |
|   9 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | apache-2.0   | 0.220      | 384  | Multilingual sentence embeddings     |
|  10 | Qdrant/clip-ViT-B-32-text                                   | mit          | 0.250      | 512  | Multimodal (text/image), English     |
|  11 | jinaai/jina-embeddings-v2-base-de                           | apache-2.0   | 0.320      | 768  | Multilingual (German-focused)        |
|  12 | BAAI/bge-base-en                                            | mit          | 0.420      | 768  | English text embeddings              |
|  13 | snowflake/snowflake-arctic-embed-m                          | apache-2.0   | 0.430      | 768  | English text embeddings              |
|  14 | thenlper/gte-base                                           | mit          | 0.440      | 768  | General-purpose embeddings           |
|  15 | jinaai/jina-embeddings-v2-base-en                           | apache-2.0   | 0.520      | 768  | English text embeddings              |
|  16 | nomic-ai/nomic-embed-text-v1                                | apache-2.0   | 0.520      | 768  | Multimodal (text/image), English     |
|  17 | nomic-ai/nomic-embed-text-v1.5                              | apache-2.0   | 0.520      | 768  | Multimodal (text/image), English     |
|  18 | snowflake/snowflake-arctic-embed-m-long                     | apache-2.0   | 0.540      | 768  | Long-context embeddings              |
|  19 | jinaai/jina-clip-v1                                         | apache-2.0   | 0.550      | 768  | Multimodal (text/image)              |
|  20 | jinaai/jina-embeddings-v2-base-code                         | apache-2.0   | 0.640      | 768  | Code + text embeddings               |
|  21 | jinaai/jina-embeddings-v2-base-zh                           | apache-2.0   | 0.640      | 768  | Chinese / mixed language             |
|  22 | jinaai/jina-embeddings-v2-base-es                           | apache-2.0   | 0.640      | 768  | Spanish / mixed language             |
|  23 | mixedbread-ai/mxbai-embed-large-v1                          | apache-2.0   | 0.640      | 1024 | Large English embeddings             |
|  24 | sentence-transformers/paraphrase-multilingual-MPNet-base-v2 | apache-2.0   | 1.000      | 768  | High-quality multilingual            |
|  25 | snowflake/snowflake-arctic-embed-l                          | apache-2.0   | 1.020      | 1024 | Large English embeddings             |
|  26 | BAAI/bge-large-en-v1.5                                      | mit          | 1.200      | 1024 | Large English embeddings             |
|  27 | thenlper/gte-large                                          | mit          | 1.200      | 1024 | Large general embeddings             |
|  28 | intfloat/multilingual-e5-large                              | mit          | 2.240      | 1024 | High-quality multilingual retrieval  |
|  29 | jinaai/jina-embeddings-v3                                   | cc-by-nc-4.0 | 2.290      | 1024 | Multi-task (retrieval/query/passage) |

## 사용 목적

### A. “검색/RAG” (쿼리 ↔ 문서 의미 매칭)

목표: semantic retrieval 품질 (Recall@k, nDCG 등)

추천 우선순위:

multilingual 지원 + retrieval용 모델

그 다음 한국어 성능 검증된 multilingual SBERT 계열

### B. “클러스터링/유사도 분석/분류용 피처”

목표: 군집/분류 성능, 안정적인 피처

추천 우선순위:

범용 sentence embedding (multilingual)

dim 너무 큰 건 오히려 운영/인덱싱 비용↑

### 컬럼 설명

size_in_GB: 실서비스에서 진짜 중요

작을수록 빠르고 배포/메모리 편함

dim: 벡터DB 비용/속도/메모리에 직결

dim ↑ → 인덱스/저장/쿼리 비용 ↑ (보통 선형에 가깝게 증가)

tasks: 보통 "embedding", "retrieval" 같은 힌트가 들어있음

"retrieval"/"search" 성격이면 RAG에 더 맞는 경우가 많음

license: 상용 서비스면 필수 체크 (Apache-2.0 / MIT 같은지)

### 선택 방법

Rule 1) 한국어면 “multilingual”을 기본값으로

한국어/영어 섞인 데이터(리뷰, 메뉴, 상호, 외래어) → multilingual이 안정적

Rule 2) 먼저 “작고 빠른 기본 모델”로 시작해서, 품질 문제 있을 때만 업그레이드

작은 모델: latency/비용 좋음 → 초기 MVP, 트래픽 대응에 강함

큰 모델: 품질은 좋아질 수 있지만, 인덱스 비용/지연/메모리 부담이 큼
→ “품질 병목이 확인됐을 때” 쓰는 게 맞음

Rule 3) dim은 “필요 이상으로 크게 하지 마”

384 / 512 / 768 / 1024 / 1536… 이런 식으로 올라가는데,

dim이 커지면:

Qdrant/HNSW 인덱스 메모리 증가

검색 속도 저하 가능

디스크/네트워크 비용 증가

보통 384~768이 운영 밸런스가 좋고,
1024 이상은 “정말 품질이 중요해서 비용 감수”할 때.

Rule 4) 같은 품질이면 “license가 더 안전한 모델”을 고르기

상용/배포 목적이면 Apache-2.0 같은 게 마음 편함.

### '프로덕션급 LLM 서비스 + Qdrant + RAG/요약' 기준, 선택 기준

Step 1: 기본 임베딩 모델 1개를 “표준”으로 고정

조건:

multilingual

size_in_GB 작은 편

dim 384~768

이걸로:

인덱싱 파이프라인

검색 평가(precision@k/recall@k)

운영 지표(P95 latency, 메모리)
를 먼저 안정화

Step 2: 품질 병목이 보이면 “상위 모델”로 A/B

같은 데이터로 top-k 성능 비교

특히 한국어에서 검색이 “느낌상” 안 맞는 경우가 많으니,
사내/도메인 쿼리셋(예: ‘가격 대비’, ‘양 많음’, ‘웨이팅’, ‘친절’, ‘위생’) 50~200개만 만들어도 차이가 바로 드러남

Step 3: dim 큰 모델을 쓰면 인덱스/저장 비용도 같이 계산

“품질 +2%” 얻으려고 “비용 2배”가 되면 손해일 수 있음

### 실무 체크리스트

각 모델을 이렇게 체크해:

description에 multilingual / cross-lingual / E5 / BGE / MiniLM / mpnet 같은 키워드가 있는가?

tasks가 retrieval/search 성격인가?

size_in_GB가 운영 환경에서 감당 가능한가? (특히 서버리스/오토스케일이면 작을수록 유리)

dim이 과하게 크지 않은가?

license가 상용 배포에 문제 없는가?

### 최고 조합 (RAG 기준)

1차 검색용 임베딩(빠름/저렴) + 2차 rerank(정확) 조합이 보통 최강이야.

임베딩은 “빠르고 적당히 좋게”

reranker(크로스 인코더/LLM judging)는 “비싸지만 top-k만”
→ 너가 이미 말한 “기본기가 받쳐줘야 RAG 품질이 오른다”랑 완전 같은 맥락.

### 바로 할만한 것

네 DF에서 size_in_GB가 작은 상위 5개를 뽑고, 그 중에서:

multilingual(또는 한국어 포함) + dim 384~768

license 안전

tasks가 retrieval/embedding

이 조건을 만족하는 1개를 “기본 모델”로 잡아.

그 다음에 size가 좀 크더라도 품질 좋다는 계열 1개를 후보로 잡고,
둘을 동일 쿼리셋으로 recall@k / precision@k 비교해서 결정.