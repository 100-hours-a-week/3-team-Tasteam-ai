### 트러블 슈팅

### 문제

스파크의 로그가 애플리케이션 동작 모니터링에 방해가 되는 문제가 발생.

운영 환경에서는 요청 흐름, 비즈니스 이벤트, 실패 지점과 같은 애플리케이션 레벨 로그를 중심으로 관측할 필요가 있음.

### 현재 SPARK 로그

WARNING: Using incubator modules: jdk.incubator.vector
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
26/01/31 20:48:40 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
[Stage 1:>                                                          (0 + 1) /

[Stage 4:>                                                          (0 + 1) /

---

### 해결 방안

Spark 로그 노이즈를 제거하고 애플리케이션 로그 중심의 관측 구조로 재정비하기 위해 아래 조치를 수행함.

콘솔 진행바 끄기 (spark.ui.showConsoleProgress=false)

log4j2 파일 명시 (spark.driver.extraJavaOptions)

Hadoop만 ERROR로 낮추기 (NativeCodeLoader 제거)

나머지(JDK incubator 경고)는 그냥 둬도 됨

### 이유

1. 콘솔 진행바는 Spark 실행 상태를 시각적으로 표현하기 위한 UI 출력

2. log4j2 설정을 통해 Spark 로그 정책을 코드와 분리하여 명시적으로 관리

3. Hadoop NativeCodeLoader 경고는 환경 특성상 반복 출력되는 노이즈 로그, 기능 영향이 없어 억제

4. JDK incubator 경고는 JVM 런타임 환경 정보를 제공하는 메시지, 추후 성능/환경 분석을 위해 유지

---

### 적용 현황

| 조치 | 적용 |
|------|------|
| 콘솔 진행바 끄기 | `spark.ui.showConsoleProgress=false` |
| log4j2 파일 명시 | `conf/log4j2-spark-quiet.properties`, `spark.driver.extraJavaOptions=-Dlog4j.configurationFile=file:...` |
| Hadoop NativeCodeLoader ERROR | `logger.nativecode.name=org.apache.hadoop.util.NativeCodeLoader`, `level=error` |

- `src/comparison_pipeline.py`: `_get_spark()`에서 위 설정 적용