
**에러**: `AttributeError: 'Qwen2ForCausalLM' object has no attribute 'set_submodule'`

**발생 위치**: `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)` 호출 시  
→ transformers 쪽 4bit 양자화에서 `model.set_submodule(module_name, new_module)` 호출  
→ PyTorch `nn.Module`에 `set_submodule`이 없어서 실패

**원인**  
`nn.Module.set_submodule()`은 **PyTorch 2.2**에서 추가된 API입니다.  
컨테이너 안의 **PyTorch가 2.2 미만**이면 (또는 2.2+가 제대로 올라오지 않았으면) 이 에러가 납니다.

- 로그 상 스택: `transformers/integrations/bitsandbytes.py` → `replace_with_bnb_linear` → `model.set_submodule(...)`
- Dockerfile 기준: 베이스 `runpod/pytorch:2.4.0-...` + `requirements.train-llm.txt`  
  → 설계상으로는 2.4여야 하지만, `pip install` 시 **torch가 다운그레이드**되거나 베이스 이미지 실제 버전이 다를 수 있음.

**대응**

1. **컨테이너에서 PyTorch 버전 확인**  
   Pod/이미지에서:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   `2.2.0` 미만이면 아래로 진행.

2. **PyTorch 2.2+ 유지**  
   `requirements.train-llm.txt`에 명시해서 재빌드:
   ```text
   torch>=2.2.0
   ```
   (이미 `requirements.train-llm.txt`에 torch가 없다면, 베이스 이미지 torch가 pip 설치 과정에서 바뀌지 않도록 `--no-deps` 등은 쓰지 말고, 필요하면 `torch>=2.2.0`을 추가해 두는 편이 안전합니다.)

3. **transformers / bitsandbytes 버전**  
   현재 요구사항(`transformers>=4.46.0`, `bitsandbytes>=0.43.0`)은 `set_submodule` 사용 코드와 맞는 편입니다.  
   컨테이너 안에서 실제 설치 버전 확인:
   ```bash
   pip show torch transformers bitsandbytes
   ```

4. **베이스 이미지**  
   `runpod/pytorch:2.4.0-...`이 실제로 PyTorch 2.4를 포함하는지 RunPod 문서/이미지 설명을 확인하고,  
   필요하면 `torch>=2.2.0`을 요구사항에 넣은 뒤 **이미지를 다시 빌드**해 sweep Pod를 새 이미지로 띄우면 됩니다.

요약하면, **이미지를 다시 빌드할 때 PyTorch가 2.2 이상이 확실히 들어가도록** (`torch>=2.2.0` 명시 + 필요 시 베이스 이미지 확인) 하면 이 에러를 피할 수 있습니다.

---


`torch>=2.2.0`을 넣었는데도 같은 에러가 나면, 아래 같은 경우일 수 있습니다.

**1. 이미지를 다시 빌드/배포하지 않음**  
- 수정한 건 **로컬의 `requirements.train-llm.txt`**뿐입니다.  
- Pod는 **이미 빌드된 Docker 이미지**를 쓰므로, **같은 내용으로 이미지를 다시 빌드하고**, 그 이미지를 쓰는 Pod를 새로 띄우지 않으면 예전 환경이 그대로입니다.

**2. 베이스 이미지의 PyTorch가 그대로 쓰이는 경우**  
- `Dockerfile.train-llm`은 `runpod/pytorch:2.4.0-...` 같은 **이미 PyTorch가 들어 있는 베이스**를 씁니다.  
- 그 다음에 `pip install -r requirements.train-llm.txt`만 하면:
  - 베이스에 이미 설치된 `torch`가 2.1 이하이면
  - `pip`는 “이미 torch가 설치되어 있다”고 보고 **업그레이드를 하지 않을 수 있습니다** (특히 `--upgrade` 없이 설치할 때).
- 그래서 **실제로 돌 때는 여전히 베이스 이미지의 오래된 torch**가 쓰이고, `set_submodule`이 없어서 에러가 날 수 있습니다.

**3. `pip install` 순서/옵션**  
- `requirements`에 `torch>=2.2.0`만 넣고,  
  `pip install --no-cache-dir -r requirements.train-llm.txt`  
  같이만 해두면, “이미 설치된 패키지는 만족하면 건드리지 않는다”는 동작 때문에 torch가 올라가지 않을 수 있습니다.

**그래서 “여전히 발생한다면” 할 수 있는 것**  
- **Dockerfile에서 torch를 명시적으로 올리기**  
  - 예: requirements 설치 전/후에  
    `RUN pip install --no-cache-dir --upgrade 'torch>=2.2.0'`  
  - 또는 requirements 설치 시  
    `pip install --no-cache-dir --upgrade -r requirements.train-llm.txt`  
  이렇게 하면 베이스에 낮은 버전이 있어도 2.2 이상으로 올라갈 가능성이 높습니다.  
- **실제 Pod/컨테이너에서 버전 확인**  
  - 그 이미지로 컨테이너를 띄운 뒤  
    `python -c "import torch; print(torch.__version__)"`  
  - 2.2 미만이면 “베이스가 오래된 torch를 쓰고 있고, pip이 올리지 않았다”는 뜻으로 보면 됩니다.

정리하면, **여전히 발생한다면**  
1) 새 이미지를 빌드해 그걸로 Pod를 띄웠는지 확인하고,  
2) 그 안에서 `torch.__version__`이 2.2 이상인지 확인한 뒤,  
3) 2.2 미만이면 Dockerfile에서 `pip install --upgrade 'torch>=2.2.0'`(또는 `--upgrade -r requirements.train-llm.txt`)를 넣어서 **강제로 2.2 이상이 깔리게** 하면 됩니다.