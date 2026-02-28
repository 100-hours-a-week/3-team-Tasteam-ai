
로그에서 **두 가지**가 보입니다.

---

**1. `ERROR - Required: pip install datasets transformers peft bitsandbytes trl`**

이 메시지는 `scripts/train_qlora.py` 96~102행의 **import 실패** 시에만 나옵니다.

```96:102:scripts/train_qlora.py
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        logger.error("Required: pip install datasets transformers peft bitsandbytes trl")
        raise SystemExit(1) from e
```

즉, **실제 원인은 위 패키지들 중 하나가 import 되지 않는 것**입니다.  
(어느 패키지에서 실패했는지는 로그에 나오지 않았지만, `e`에 담겨 있을 것입니다.)

가능한 이유:
- Pod에서 쓰는 **이미지에 위 의존성이 제대로 설치되지 않음** (다른 이미지를 쓰거나, `requirements.train-llm.txt` 설치가 실패/생략된 빌드).
- **trl** 등이 optional dependency 때문에 일부 환경에서만 ImportError를 내는 경우.

**대응:**
- Pod와 동일한 이미지로 컨테이너를 띄운 뒤, 그 안에서  
  `python -c "from datasets import Dataset; from transformers import ...; from peft import ...; from trl import SFTConfig, SFTTrainer"`  
  를 실행해 **어느 import에서 실패하는지** 확인.
- 실패한 패키지를 `requirements.train-llm.txt` / Dockerfile에서 명시하고, **이미지 다시 빌드** 후 같은 이미지로 sweep Pod를 띄우기.

---

**2. `wandb.sdk.lib.asyncio_manager.AlreadyJoinedError`**

wandb 내부 스레드(`IntMsgThr`, `ChkStopThr`)에서 `_schedule` 호출 시 나는 오류입니다.  
sweep agent가 **한 프로세스에서 run을 연속으로** 돌릴 때, asyncio 이벤트 루프가 run 간에 완전히 정리되지 않으면 발생할 수 있습니다.

**대응:**
- wandb 업그레이드: `pip install --upgrade wandb` (이미지 재빌드 시).
- 또는 sweep 시 **run당 프로세스 분리**: 예를 들어 `wandb.agent(..., count=1)`만 쓰고, agent를 여러 번 띄우는 방식으로 한 run당 한 프로세스가 되게 하면, 이 오류 가능성이 줄어듭니다.

---

**정리**

- **학습이 아예 안 되고 바로 “Required: pip install …”만 나온다** → 위 import가 실패하는 쪽이 근본 원인입니다. Pod 이미지에 `datasets`, `transformers`, `peft`, `bitsandbytes`, `trl`이 제대로 설치된 이미지로 다시 빌드하고, 그 이미지에서 어느 import가 실패하는지 확인하는 것이 좋습니다.
- **이미 학습은 되는데 로그에만 AlreadyJoinedError가 찍힌다** → wandb asyncio 이슈이므로, wandb 업그레이드나 run당 프로세스 분리로 완화할 수 있습니다.

원하시면 Agent 모드에서 `train_qlora.py`에 **실제 예외 메시지(`e`)를 로그에 남기도록** 수정해, 다음에 같은 로그가 나왔을 때 어떤 패키지에서 import가 깨졌는지 바로 보이게 할 수 있습니다.