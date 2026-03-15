
이번에는 `--eval-timeout 28800`으로 플로우가 **완료**됐지만, 결과가 여전히 **옛날 버전**을 가리키고 있습니다.

- **Result**: `report_path`·`artifact_version`·`eval_dir` 모두 **20260303_053420**
- 이번에 Pod가 만든 eval은 `eval_output/20260312_xxxxx/` 같은 **새 버전**이어야 하는데, 그쪽이 아니라 기존 `eval/20260303_053420`이 쓰인 상태입니다.

즉, 예전에 말한 **“다운로드는 됐는데 rglob이 다른 report를 골라서 잘못된 버전이 쓰이는”** 문제가 그대로 있는 겁니다.  
다운로드 시 **버전 서브디렉터리**(예: `out_dir/20260312_xxxxx/report.json`)에 받고, 그 경로만 쓰도록 수정해야 새로 받은 eval이 결과에 반영됩니다. 원하면 그 수정 포인트까지 짚어서 패치 방향 정리해 줄게요.

---


네. **그럴 가능성이 큽니다.**

플로우가 정상적으로 끝나면:

1. `eval_done.json` 확인 후  
2. `download_directory_from_runpod(vol_id, eval_output_prefix, out_dir)` 로 해당 버전을 받고  
3. 이어서 **`delete_prefix_from_volume(vol_id, eval_output_prefix)`** 로 그 prefix를 **볼륨에서 삭제**합니다.

관련 코드는 `scripts/distill_flows.py` 1564–1575 근처입니다:

```python
n_files = download_directory_from_runpod(vol_id, eval_output_prefix, out_dir)
...
try:
    deleted = delete_prefix_from_volume(vol_id, eval_output_prefix)
```

그래서 **한 번이라도 evaluate_on_pod가 “성공”으로 끝났다면**,  
그때 받은 `eval_output/<version>/` 은 이미 **다운로드 후 삭제**된 상태입니다.  
그래서 지금 `volume_ls --prefix distill_pipeline_output/eval_output` 이 비어 있는 게, “설마 삭제 로직으로 삭제된 건가?” 하는 추측과 맞습니다.

다만 그때는 **다운로드는 됐지만** `out_dir` 에 버전 폴더 없이 받아져서 `rglob("report.json")` 이 기존 `eval/20260303_053420` 을 골라 버린 것이고,  
**실제 새로 받은 파일**은 `distill_pipeline_output/` 루트에 `report.json` 등으로 있을 수 있습니다.  
그 파일들이 아직 있다면, 그걸 버전 폴더로 옮겨서 쓰거나, 다음부터는 “버전 서브디렉터리에 받고 그 경로만 쓰는” 수정을 하면 됩니다.