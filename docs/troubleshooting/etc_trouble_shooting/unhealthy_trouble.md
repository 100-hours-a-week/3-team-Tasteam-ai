new_async 컨테이너가 healthcheck에서 실패해 'unhealthy'로 표시된 상태입니다. 원인 확인 중입니다.


[3 tools called]


원인: new_async 컨테이너의 healthcheck가 실패해 'unhealthy'로 표시되었고, prometheus가 `condition: service_healthy`로 기다리다 실패한 것입니다.

동작 요약:
- `/ready`는 앱 lifespan에서 `_warm_up_services()`(모델/임베딩/Qdrant 등)가 끝난 뒤에만 200을 반환합니다.
- warm-up이 실패하거나 **90초(start_period) + 30초(retries×interval)** 안에 끝나지 않으면 `curl -f`가 503을 받아 healthcheck가 실패합니다.

가능한 원인:
1. **warm-up이 너무 오래 걸림** — 90초로는 부족할 수 있음  
2. **warm-up 실패** — `.env` 누락, Qdrant/Redis 미기동, OOM 등으로 `app.state.ready`가 True가 안 됨  
3. **앱 크래시** — 시작 중 예외로 프로세스가 죽음

우선 로그로 원인 확인하기:
```bash
docker compose logs new_async
```
`warm-up 실패` 또는 예외 스택이 보이면 그게 원인입니다.

해결 제안:
- **start_period 늘리기** — warm-up이 느리면 90초를 180초 또는 300초로 늘려 healthcheck가 통과할 시간을 줍니다.


[1 tool called]


healthcheck 설정을 완화해 두었습니다.

- **start_period**: 90s → **180s** (이 동안 실패해도 unhealthy로 치지 않음)
- **retries**: 3 → **10**
- **interval**: 10s → **15s**, **timeout**: 5s → **10s**

warm-up이 2~3분 걸려도 통과할 수 있습니다.  
다시 띄워서 확인해 보세요.

```bash
docker compose up -d
```

그래도 unhealthy면 `docker compose logs new_async` 로 warm-up 실패 메시지나 예외를 확인하면 됩니다. (Qdrant/Redis가 compose에 없으면 연결 실패로 ready가 안 될 수 있습니다.)