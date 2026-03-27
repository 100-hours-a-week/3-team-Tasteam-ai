# Distill 2-Step Runbook

`distill-orchestrator` 컨테이너에서 아래 2단계를 순차 실행합니다.

## 1) Sweep + Best Adapter 다운로드

```bash
make distill-sweep-best \
  DISTILL_LABELED_PATH=distill_pipeline_output/labeled/<version>/train_labeled.json \
  DISTILL_OUT_DIR=distill_pipeline_output \
  DISTILL_NUM_PODS=2
```

선택:

- `DISTILL_SWEEP_ID=<entity/project/sweep_id>` 지정 시 기존 sweep 재사용

## 2) Evaluate on Pod

```bash
make distill-eval-pod \
  DISTILL_ADAPTER_PATH=distill_pipeline_output/artifacts/<run_id>/adapter \
  DISTILL_VAL_PATH=distill_pipeline_output/labeled/<version>/val_labeled.json \
  DISTILL_TEST_PATH=distill_pipeline_output/labeled/<version>/test_labeled.json \
  DISTILL_OUT_DIR=distill_pipeline_output
```

## 산출물 경로

- Best adapter: `distill_pipeline_output/artifacts/<run_id>/adapter`
- Eval report: `distill_pipeline_output/eval/.../report.json`
- LLM judge: `distill_pipeline_output/eval/.../llm_as_a_judge_results.json`
- KD/SFT analysis: `distill_pipeline_output/eval/.../kd_sft_analysis_report.json`

## 참고

- distill 컨테이너는 named volume `tasteam_distill-output`을 사용합니다.
- 필수 env: `RUNPOD_API_KEY`, `RUNPOD_S3_ACCESS_KEY`, `RUNPOD_S3_SECRET_ACCESS_KEY`, `WANDB_API_KEY`, `WANDB_PROJECT`
