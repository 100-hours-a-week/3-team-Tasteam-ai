
로그에 나온 에러는 다음 한 줄로 정리됩니다.

**`ModuleNotFoundError: No module named 'comparison_lib'`**

---

## 원인

- **드라이버(Spark 서비스 프로세스)** 는 `scripts/spark_service.py`에서 `PROJECT_ROOT`를 `sys.path`에 넣어서 `comparison_lib`를 불러올 수 있음.
- **Spark가 기동하는 Python 워커** 는 별도 서브프로세스라서, 드라이버의 `sys.path`를 물려받지 않음.
- `mapPartitions` 등으로 전달된 함수가 **cloudpickle로 직렬화**될 때 `comparison_lib`에 대한 참조가 들어가고, 워커에서 이걸 복원하려다 `import comparison_lib`가 실행되면서 **워커 쪽에는 `comparison_lib`가 없어서** 위 에러가 남.

즉, **드라이버에는 있고 워커에는 없는 경로/모듈** 때문에 생기는 문제입니다.

---

## 해결 방향

워커 프로세스가 `comparison_lib`를 찾을 수 있게 **워커가 쓰는 Python 환경의 `PYTHONPATH`에 프로젝트 루트(/app)를 넣어주면** 됩니다.

1. **Dockerfile.spark-service**  
   - `ENV PYTHONPATH=/app`  
   - 또는 `ENV PYTHONPATH=/app:$PYTHONPATH`  
   이렇게 두면 컨테이너 안에서 실행되는 **모든** Python 프로세스(드라이버 + Spark가 띄우는 워커)가 `PYTHONPATH`를 물려받아 `import comparison_lib`가 동작합니다.

2. **또는** Spark 설정으로 워커 환경에만 넣고 싶다면,  
   - `SparkSession.builder.config("spark.executorEnv.PYTHONPATH", "/app")`  
   처럼 executor(워커) 환경 변수에 `PYTHONPATH`를 지정하는 방법도 있습니다.  
   (실제 경로는 이미지 안 프로젝트 루트에 맞춰서.)

정리하면, **Spark 워커가 `comparison_lib`를 못 찾아서 500이 나는 상황**이므로, 위처럼 **워커가 쓰는 환경에 `PYTHONPATH=/app`(또는 프로젝트 루트)를 넣어주면** 해결됩니다.