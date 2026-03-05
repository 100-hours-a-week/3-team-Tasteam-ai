
원인은 **컨테이너 안에서 Java 경로가 안 맞는 것**입니다.

---

## 로그가 말하는 것

- ` /usr/lib/jvm/java-17-openjdk-amd64/bin/java: No such file or directory`  
  → PySpark가 기대하는 Java 경로에 실제로 `java`가 없음.
- Dockerfile에는 `JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64` 로 **amd64 경로를 고정**해 두었음.

그래서 **아키텍처가 amd64가 아닐 때** (예: Mac M1/M2에서 arm64 이미지로 빌드/실행) Debian은 보통  
`/usr/lib/jvm/java-17-openjdk-arm64`  
처럼 설치하고, `java-17-openjdk-amd64` 디렉터리는 없습니다.  
→ Java를 찾지 못하고 JVM이 안 떠서 500이 나는 상황입니다.

---

## 해결 방향

**`JAVA_HOME`을 아키텍처에 맞는 실제 경로로 맞춰주면 됩니다.**

1. **Dockerfile에서 경로를 고정하지 않고 잡기**  
   - 예: `RUN`에서 `which java` / `readlink` 로 실제 JVM 디렉터리를 구한 뒤,  
     그 경로를 쓰는 `ENV JAVA_HOME=...` 를 설정하거나,  
   - 또는 `RUN`에서  
     `ln -s $(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1) /usr/lib/jvm/java-17-openjdk-amd64`  
     처럼 현재 아키텍처용 디렉터리를 `java-17-openjdk-amd64` 로 심링크해서, 지금처럼 `JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64` 를 써도 동작하게 할 수 있습니다.

2. **실행 시점에만 맞추기**  
   - `docker run` 할 때  
     `-e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64`  
     처럼 실제 있는 경로를 넘겨서 테스트해 볼 수 있습니다. (arm64면 위 경로가 맞는지 컨테이너 안에서 `ls /usr/lib/jvm/` 로 확인)

정리하면, **리소스 제한 때문이 아니라 “Spark 서비스 컨테이너 안에서 Java 경로(JAVA_HOME)가 아키텍처와 안 맞아서”** 500이 나는 상황이고, Dockerfile이나 `docker run`에서 `JAVA_HOME`을 실제 Java가 설치된 경로로 맞춰주면 해결됩니다.