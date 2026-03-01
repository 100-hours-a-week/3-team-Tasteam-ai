config has no attribute use_runpod error

---

prefect_long_analysis/long_logs/config_has_no_attribute_use_runpod.logs 참조

---

use_runpod 주석 해제 (true)

---

success

proxy_success_prefect_logs.log

16:33:43.500 | INFO    | Task run 'labeling-pod-only-task-3a0' - Cleaning up pod: 9i28c9dokswe5f
16:33:44.729 | INFO    | Task run 'labeling-pod-only-task-3a0' - Finished in state Completed()
16:33:44.763 | INFO    | Flow run 'elated-chameleon' - Finished in state Completed()
Result: {'labeled_path': 'distill_pipeline_output/labeled/20260226_051037/train_labeled.json', 'val_labeled_path': 'distill_pipeline_output/labeled/20260226_051037/val_labeled.json', 'test_labeled_path': 'distill_pipeline_output/labeled/20260226_051037/test_labeled.json', 'runpod_upload': {'uploaded': True, 'count': 4, 'labeled_dir': 'distill_pipeline_output/labeled/20260226_051037'}}
16:33:44.769 | INFO    | prefect - Stopping temporary server on http://127.0.0.1:8172