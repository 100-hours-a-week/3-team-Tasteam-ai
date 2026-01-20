1. API 엔드포인트 수정
    1. 기존 방식 : 
        1. 1차로 인코더 모델이 분류해 긍/부정 개수 반환
        2. 인코더 모델의 threshold가 낮거나, 인코더 모델이 제대로 분류하기 힘든 리뷰의 경우 llm이 그 해당 전체 리뷰에서 긍/부정 개수 반환
        3. 추후 취합해
        4. 긍/부정 count, 긍/부정 비율을 반환하는 방식
    2. 바뀐 방식 : 
    Qwen2.5-7b-instruct의 Context Length: Full 131,072 tokens and generation 8192 tokens의 특성과 GPU의 빠른 추론 속도, vllm의 병렬처리의 강점을 이용해 전체 리뷰를 배치 방식으로 넣고 생성 진행해 llm이 한번에 최종 결과인 긍/부정 개수 반환하고 긍/부정 비율 계산.
2. 아키텍쳐 수정
    1. 기존 :
        1. vectordb → cpu서버, llm → gpu 서버
    2. 바뀐 방식 :
        1. vectordb(qdrant)는 on-disk 방식
        2. llm은 vllm runpod 서버리스 엔드포인트 방식
    3. vllm 서빙을 택한 이유
        1. runpod에서 vllm endpoint template을 통해 서빙의 편의를 제공해줬다.
        2. runpod의 이용 목적은 서빙이었는데, 기존엔 runpod severless endpoint에 허깅페이스 모델들을 사용할려면 docker image에 미리 빌드해서 runpod serverless에 올렸어야 함. 빌드 시간이 너무 오래걸리고, 빌드에 성공했는데도, 그게 빌드가 잘되지 않아 runpod에서 도커 이미지가 정상 작동 하지 않았으면 다시 빌드해야하고, 따라서 시간이 너무 오래걸렸음. 하지만, vllm 자체적으로 허깅페이스 모델을 지원해 서빙 편의가 매우 향상됐음.
3. 추가 내용
    1. 모델
        1. qwen2.5-7b-instruct
    2. 모델 선택 이유
        1. 많은 한국어 학습 데이터로 학습이 되있음.
        2. 3b 이하로 하기엔 추론 성능이 너무 안좋을거 같고, GPU 환경이더라도 14b는 너무 무거워서 큰 리소스로 인해, 추론 시간, 리소스 소모 측면에서 문제 발생 가능성. 그리고, 그 정도로 좋은 성능이 필요한 task들이 아님.
        3. Instruct 모델의 선택 이유
            1. instruct의 ‘사람의 지시를 이해하는 능력’을 통해 task 명령시 더 사람의 의도를 잘 파악하고 따름.
    3. 모델 스펙
        - Type: Causal Language Models
        - Training Stage: Pretraining & Post-training
        - Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
        - Number of Parameters: 7.61B
        - Number of Paramaters (Non-Embedding): 6.53B
        - Number of Layers: 28
        - Number of Attention Heads (GQA): 28 for Q and 4 for KV
        - Context Length: Full 131,072 tokens and generation 8192 tokens
        - 출처: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
    - Qwen2.5-7b-instruct vs qwen3-4b-instruct
        - qwen3-4b-instruct의 한계
            - 가볍지만 추론 성능이 약함.
                - 하지만 서비스의 task들은 추론 성능을 필요로 함.
            - 간단한 지시, 단일 문장 긍/부정, 리뷰 전체에 대해서로만 판단, 각 문장을 분해하는 능력이 없음.
        - 따라서 qwen2.5-7b-instruct 선택.
            - 해당 tasks들을 원활히 수행 가능한 qwen2.5-7b-instruct 선택.