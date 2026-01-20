## 최적화 적용

### 1. vLLM 기반 LLM 추론 최적화
- **Continuous Batching**: vLLM이 여러 배치를 자동으로 효율적으로 처리 (GPU 활용률 극대화)
- **PagedAttention**: KV Cache를 페이지 단위로 관리 (vLLM 내장, 메모리 사용량 최적화)
- **Tensor Parallel**: 다중 GPU 환경에서 모델 분산 (선택적, 기본값: 1)
- **로컬 추론**: RunPod Pod 환경에서 vLLM 직접 사용 (네트워크 오버헤드 제거)

### 2. 임베딩 모델 최적화 (SentenceTransformer)
- **GPU 활용**: CUDA를 통한 병렬 연산
- **FP16 양자화**: 메모리 50% 절감 및 연산 속도 향상
- **배치 처리**: 여러 텍스트를 한 번에 처리

### 3. 동적 배치 크기 최적화
- **리뷰 길이 기반 계산**: 평균 토큰 수 추정 후 배치 크기 자동 조정
- **최대 토큰 수 제한**: `VLLM_MAX_TOKENS_PER_BATCH` (기본값: 4000)
- **최소/최대 제한**: `VLLM_MIN_BATCH_SIZE` (10) ~ `VLLM_MAX_BATCH_SIZE` (100)

### 4. OOM 방지 메커니즘
- **세마포어 제한**: 동시 처리 배치 수 제한 (`VLLM_MAX_CONCURRENT_BATCHES`: 20)
- **동적 배치 크기 + 세마포어**: 단일 배치 및 누적 메모리 모두 제어

### 5. 비동기 처리 최적화
- **ThreadPoolExecutor**: vLLM의 동기 API를 비동기로 변환
- **asyncio.gather**: 여러 배치를 병렬로 처리

### 6. 로컬 모델 최적화 (선택적, USE_POD_VLLM=false일 때)
- **Flash Attention-2**: 메모리 효율 및 긴 시퀀스 처리 속도 향상 (선택적)
- **FP16 양자화**: 메모리 절감 및 속도 향상
- **device_map="auto"**: 모델을 자동으로 GPU에 분산 로드

### 7. GPU 메모리 기반 배치 크기 최적화
- **GPU 메모리에 따른 동적 조정**: A100 (40GB+) / RTX 3090 (24GB) / 기타 자동 조정

---

**모델 추론 측정 지표**
1. latency (레이턴시)
2. Throughput (처리량)
3. GPU 사용률
4. 병목 분석

추후 최적화 실험 추가.

---

## 추가 최적화 방안

### 단기 (검토 중)
- 더 큰 배치 크기 사용 (GPU 메모리 여유 시)
- 비동기 임베딩 처리

### 장기 (필요 시)
- 모델 교체 (더 작고 빠른 모델)
- 지식 증류 (모델 경량화)
- 양자화 변경 (모델 경량화)




