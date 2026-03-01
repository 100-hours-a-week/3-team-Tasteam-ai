# wandb sweep OOM 오류 정리 (inner_training_loop.log)

로그를 기준으로 보면 다음 세 가지가 반복되고 있습니다.

---

## 1. Eval 단계에서 OOM (가장 빈번)

- **발생**: step 100 근처 첫 eval 시
- **메시지**: `Tried to allocate 9.27 GiB. GPU 0 has a total capacity of 23.52 GiB of which 7.48 GiB is free.`
- **원인**: eval 시 gradient checkpointing 미적용, 기본 batch_size로 학습과 동일하게 사용 → 추가 ~9GB 필요

**적용한 대응**: `per_device_eval_batch_size=1` 설정 (train_qlora.py)

---

## 2. 이전 run OOM 이후 "Some modules dispatched on CPU"

- **발생**: OOM 후 같은 프로세스에서 다음 run 시작 시 `from_pretrained(..., device_map="auto")`
- **원인**: GPU 메모리 비정상/파편화 → device_map이 일부 CPU로 보냄, 4bit BNB는 CPU 오프로드 미지원

**적용한 대응**: run_train 시작 시 `gc.collect()` + `torch.cuda.empty_cache()` (train_qlora.py)

---

## 3. batch_size=4 + max_seq_length=2048 조합 OOM

- **발생**: 학습 첫 스텝에서 `22.35 GiB allocated`, 24GB 초과

**적용한 대응**: sweep에서 `batch_size`를 `[1, 2]`로 제한 (wandb_sweep_qlora.yaml)

---

## 적용 요약

| 원인 | 적용 |
|------|------|
| Eval OOM | per_device_eval_batch_size=1 |
| 연쇄 실패 (CPU 디스패치) | run_train 시작 시 gc + empty_cache |
| batch_size 4 OOM | batch_size values: [1, 2] |
