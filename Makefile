COMPOSE_BASE = docker compose -f compose.base.yml
COMPOSE_STACK = $(COMPOSE_BASE) -f compose.stack.yml
COMPOSE_DEV = docker compose -f docker-compose.yml
DISTILL_RUN = $(COMPOSE_STACK) run --rm distill-orchestrator

# Distill defaults (필요 시 make 실행 시점에 override)
# 예: make distill-sweep-best DISTILL_LABELED_PATH=distill_pipeline_output/labeled/20260322_120000/train_labeled.json
DISTILL_OUT_DIR ?= distill_pipeline_output
DISTILL_NUM_PODS ?= 2
DISTILL_SWEEP_ID ?=
DISTILL_LABELED_PATH ?= distill_pipeline_output/labeled/REPLACE_ME/train_labeled.json
DISTILL_ADAPTER_PATH ?= distill_pipeline_output/artifacts/REPLACE_ME/adapter
DISTILL_VAL_PATH ?= distill_pipeline_output/labeled/REPLACE_ME/val_labeled.json
DISTILL_TEST_PATH ?= distill_pipeline_output/labeled/REPLACE_ME/test_labeled.json

.PHONY: help \
	up-dev down-dev logs-dev ps-dev \
	up-app down-app logs-app \
	up-batch down-batch logs-batch \
	up-obs down-obs logs-obs \
	up-ml down-ml logs-ml \
	up-all down-all ps \
	up-app-split up-batch-split up-obs-split up-ml-split \
	distill-sweep-best distill-eval-pod distill-two-step

help:
	@echo "Usage:"
	@echo "  make up-dev        # docker-compose.yml (dev 통합)"
	@echo "  make up-app        # profile=app"
	@echo "  make up-batch      # profile=batch"
	@echo "  make up-obs        # profile=obs"
	@echo "  make up-ml         # profile=ml"
	@echo "  make up-all        # app+batch+obs+ml 전체"
	@echo "  make down-all      # 전체 중지"
	@echo "  make ps            # compose.stack 상태"
	@echo ""
	@echo "Distill two-step:"
	@echo "  make distill-sweep-best  # sweep_pod_best_adapter 실행"
	@echo "  make distill-eval-pod    # evaluate_on_pod 실행"
	@echo "  make distill-two-step    # 위 두 단계 연속 실행"
	@echo ""
	@echo "Split compose files:"
	@echo "  make up-app-split   # compose.app.yml"
	@echo "  make up-batch-split # compose.batch.yml"
	@echo "  make up-obs-split   # compose.obs.yml"
	@echo "  make up-ml-split    # compose.ml.yml"

up-dev:
	$(COMPOSE_DEV) up -d --build

down-dev:
	$(COMPOSE_DEV) down

logs-dev:
	$(COMPOSE_DEV) logs -f

ps-dev:
	$(COMPOSE_DEV) ps

up-app:
	$(COMPOSE_STACK) --profile app up -d --build

down-app:
	$(COMPOSE_STACK) --profile app down

logs-app:
	$(COMPOSE_STACK) logs -f app retrieval-service

up-batch:
	$(COMPOSE_STACK) --profile batch up -d --build

down-batch:
	$(COMPOSE_STACK) --profile batch down

logs-batch:
	$(COMPOSE_STACK) logs -f batch-worker redis spark-service

up-obs:
	$(COMPOSE_STACK) --profile obs up -d --build

down-obs:
	$(COMPOSE_STACK) --profile obs down

logs-obs:
	$(COMPOSE_STACK) logs -f prometheus grafana alertmanager jobmgr node-exporter

up-ml:
	$(COMPOSE_STACK) --profile ml up -d --build

down-ml:
	$(COMPOSE_STACK) --profile ml down

logs-ml:
	$(COMPOSE_STACK) logs -f deepfm-training deepfm-inference distill-orchestrator

up-all:
	$(COMPOSE_STACK) --profile app --profile batch --profile obs --profile ml up -d --build

down-all:
	$(COMPOSE_STACK) down

ps:
	$(COMPOSE_STACK) ps

up-app-split:
	$(COMPOSE_BASE) -f compose.app.yml up -d --build

up-batch-split:
	$(COMPOSE_BASE) -f compose.batch.yml up -d --build

up-obs-split:
	$(COMPOSE_BASE) -f compose.obs.yml up -d --build

up-ml-split:
	$(COMPOSE_BASE) -f compose.ml.yml up -d --build

distill-sweep-best:
	$(DISTILL_RUN) sweep_pod_best_adapter \
		--labeled-path $(DISTILL_LABELED_PATH) \
		--out-dir $(DISTILL_OUT_DIR) \
		--num-pods $(DISTILL_NUM_PODS) \
		$(if $(DISTILL_SWEEP_ID),--sweep-id $(DISTILL_SWEEP_ID),)

distill-eval-pod:
	$(DISTILL_RUN) evaluate_on_pod \
		--adapter-path $(DISTILL_ADAPTER_PATH) \
		--val-labeled-path $(DISTILL_VAL_PATH) \
		$(if $(DISTILL_TEST_PATH),--test-labeled-path $(DISTILL_TEST_PATH),) \
		--out-dir $(DISTILL_OUT_DIR)

distill-two-step: distill-sweep-best distill-eval-pod
