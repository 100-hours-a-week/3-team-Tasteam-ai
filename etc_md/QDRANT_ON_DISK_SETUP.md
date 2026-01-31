# Qdrant On-Disk 설정 완료

## 변경 사항

### 1. ✅ 기본 설정 변경

**파일**: `src/config.py`

**변경 내용**:
- 기본값을 `:memory:`에서 `./qdrant_db`로 변경
- on-disk 모드가 기본값으로 설정됨

```python
# 변경 전
QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", ":memory:")

# 변경 후
QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", "./qdrant_db")
```

---

### 2. ✅ 디렉토리 자동 생성

**파일**: `src/api/dependencies.py`

**변경 내용**:
- on-disk 모드 사용 시 디렉토리 자동 생성
- 상대 경로를 절대 경로로 변환 (프로젝트 루트 기준)

```python
# 디렉토리 자동 생성
os.makedirs(qdrant_path, exist_ok=True)
```

---

### 3. ✅ 환경 변수 설정 추가

**파일**: `.env.example`

**추가 내용**:
```bash
# Qdrant 설정
QDRANT_URL="./qdrant_db"
```

---

### 4. ✅ .gitignore 업데이트

**파일**: `.gitignore`

**추가 내용**:
```
# Qdrant on-disk 데이터베이스
qdrant_db/
*.qdrant
```

---

## 사용 방법

### 1. 기본 사용 (on-disk 모드)

환경 변수를 설정하지 않으면 자동으로 `./qdrant_db` 디렉토리에 저장됩니다.

```bash
# 서버 실행
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

### 2. 환경 변수로 설정

`.env` 파일에 설정:

```bash
# on-disk 모드 (기본값)
QDRANT_URL="./qdrant_db"

# 또는 절대 경로
QDRANT_URL="/path/to/qdrant_db"

# 메모리 모드 (테스트용)
QDRANT_URL=":memory:"

# 원격 서버
QDRANT_URL="http://localhost:6333"
```

### 3. 다른 디렉토리 사용

```bash
# 환경 변수로 설정
export QDRANT_URL="./data/qdrant_db"

# 또는 .env 파일에 추가
echo 'QDRANT_URL="./data/qdrant_db"' >> .env
```

---

## Qdrant 모드 비교

### 1. On-Disk 모드 (기본값)
- **장점**: 데이터 영속성, 서버 재시작 후에도 데이터 유지
- **단점**: 디스크 I/O로 인한 약간의 성능 저하
- **사용 사례**: 프로덕션 환경, 장기 데이터 저장

### 2. 메모리 모드
- **장점**: 빠른 성능, 테스트에 적합
- **단점**: 서버 재시작 시 데이터 삭제
- **사용 사례**: 테스트, 개발 환경

### 3. 원격 서버 모드
- **장점**: 분산 환경, 확장성
- **단점**: 네트워크 지연
- **사용 사례**: 마이크로서비스 아키텍처, 클러스터 환경

---

## 데이터 위치

### 기본 위치
```
프로젝트 루트/
  └── qdrant_db/          # Qdrant 데이터베이스
      ├── collections/     # 컬렉션 데이터
      ├── snapshots/      # 스냅샷
      └── ...
```

### 커스텀 위치
환경 변수 `QDRANT_URL`로 원하는 경로 지정 가능

---

## 주의사항

### 1. 디스크 공간
- 벡터 데이터는 크기가 클 수 있음
- 충분한 디스크 공간 확보 필요

### 2. 백업
- on-disk 모드에서는 `qdrant_db/` 디렉토리를 백업하면 됨
- 정기적인 백업 권장

### 3. Git
- `qdrant_db/` 디렉토리는 `.gitignore`에 추가되어 커밋되지 않음
- 프로덕션 환경에서는 별도 백업 전략 필요

### 4. 권한
- Qdrant 디렉토리에 읽기/쓰기 권한 필요
- 디렉토리 생성 권한 필요

---

## 마이그레이션

### 메모리 모드에서 On-Disk 모드로 전환

1. **데이터 내보내기** (메모리 모드에서):
```python
# 컬렉션 데이터 내보내기
from qdrant_client import QdrantClient
client = QdrantClient(location=":memory:")
# 스냅샷 생성 또는 데이터 내보내기
```

2. **On-Disk 모드로 전환**:
```bash
# 환경 변수 설정
export QDRANT_URL="./qdrant_db"
```

3. **데이터 가져오기** (필요한 경우):
```python
# 컬렉션 데이터 가져오기
from qdrant_client import QdrantClient
client = QdrantClient(path="./qdrant_db")
# 스냅샷 복원 또는 데이터 가져오기
```

---

## 검증

### 설정 확인

```python
from src.config import Config
print(f"QDRANT_URL: {Config.QDRANT_URL}")
```

### 디렉토리 확인

```bash
# 디렉토리 생성 확인
ls -la qdrant_db/

# 또는
ls -la ./qdrant_db/
```

---

## 성능 최적화

### 1. SSD 사용
- SSD에 저장하면 성능 향상

### 2. 디렉토리 위치
- 빠른 디스크에 저장
- 네트워크 디스크는 피하는 것이 좋음

### 3. 백업 전략
- 정기적인 스냅샷 생성
- 증분 백업 고려

---

## 문제 해결

### 디렉토리 생성 실패
```
PermissionError: [Errno 13] Permission denied
```

**해결책**:
```bash
# 디렉토리 권한 확인
ls -la | grep qdrant_db

# 권한 부여
chmod 755 qdrant_db/
```

### 디스크 공간 부족
```
OSError: [Errno 28] No space left on device
```

**해결책**:
- 디스크 공간 확보
- 다른 위치로 변경: `export QDRANT_URL="/path/to/larger/disk/qdrant_db"`

---

## 완료

✅ Qdrant 기본 설정을 on-disk 모드로 변경
✅ 디렉토리 자동 생성 기능 추가
✅ 환경 변수 설정 예시 추가
✅ .gitignore 업데이트

이제 Qdrant 데이터가 디스크에 영구 저장됩니다.
