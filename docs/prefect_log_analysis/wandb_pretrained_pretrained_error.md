
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