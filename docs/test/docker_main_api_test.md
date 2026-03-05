
가능합니다. 흐름만 정리해 보겠습니다.

---

## 1. 이미지 빌드

```bash
# 메인 API (현재 dockerfile 기준)
docker build -t main-api:latest -f dockerfile .

# Spark 서비스
docker build -t spark-service:latest -f Dockerfile.spark-service .
```

docker-compose에서 쓰는 이름이 `tasteam-new-async:latest`라면, 그걸 쓰려면:
`docker build -t tasteam-new-async:latest -f dockerfile .`  
원하시는 태그(`main-api:latest` vs `tasteam-new-async:latest`)에 맞춰 하나만 쓰면 됩니다.

---

## 2. 네트워크 생성 (컨테이너끼리 통신용)

```bash
docker network create testnet
```

---

## 3. Spark 서비스만 켜기 (RAM 측정용 – Spark 미사용)

```bash
docker run -d --name main-api --network testnet -p 8001:8001 --env-file .env \
  -e SPARK_SERVICE_URL= \
  main-api:latest
```

- `SPARK_SERVICE_URL=` 비움 → 메인 API만 Kiwi 사용, Spark 서비스 미사용.
- 다른 터미널에서: `docker stats main-api` 로 RAM 확인.

---

## 4. Spark 서비스 사용 상태로 둘 다 켜기

```bash
# 1) Spark 서비스 먼저 (같은 네트워크, 이름 spark-service)
docker run -d --name spark-service --network testnet -p 8002:8002 \
  -e SPARK_SERVICE_URL= \
  spark-service:latest

# 2) 메인 API (Spark 서비스 URL 지정)
docker run -d --name main-api --network testnet -p 8001:8001 --env-file .env \
  -e SPARK_SERVICE_URL=http://spark-service:8002 \
  main-api:latest
```

- `main-api`는 `http://spark-service:8002` 로 Spark 서비스 호출.
- `docker stats main-api spark-service` 로 두 컨테이너 RAM 동시 측정.

---

## 5. 테스트 요청 보내기

서버 기동 후 (ready 될 때까지 잠시 대기):

```bash
BASE_URL=http://localhost:8001 python test_all_task.py
```

원하면 `--tests sentiment summarize comparison` 처럼 일부만 실행해도 됩니다.

---

## 6. RAM 측정

```bash
# 두 컨테이너 모두
docker stats main-api spark-service

# 또는 전체
docker stats
```

- **Spark 미사용**: `main-api` 하나만 실행한 뒤 위처럼 테스트 + `docker stats main-api`.
- **Spark 사용**: `main-api` + `spark-service` 둘 다 띄운 뒤 테스트 + `docker stats main-api spark-service`.

이렇게 하면 “main-api만” vs “main-api + spark-service 사용” 상태에서 각각 RAM을 측정할 수 있습니다.