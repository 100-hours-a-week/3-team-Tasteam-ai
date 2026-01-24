#!/bin/bash
set -e

APP_DIR="/home/appuser/ai"
PORT=8001

echo "===== AI 서버 배포 시작 ====="

# 1. 기존 프로세스 중지 (포트 기반)
echo "[1/5] 기존 프로세스 종료"
PID=$(lsof -ti :$PORT || true)
if [ -n "$PID" ]; then
  echo "기존 프로세스 종료 중 (PID: $PID)..."
  kill -15 "$PID"

  # 프로세스가 완전히 종료될 때까지 대기 (최대 10초)
  for i in {1..10}; do
    if ! lsof -ti :$PORT > /dev/null 2>&1; then
      echo "기존 프로세스 종료 완료"
      break
    fi
    echo "프로세스 종료 대기 중... ($i/10)"
    sleep 1
  done

  # 여전히 실행 중이면 강제 종료
  if lsof -ti :$PORT > /dev/null 2>&1; then
    echo "강제 종료 실행 중..."
    kill -9 $(lsof -ti :$PORT)
    sleep 1
  fi
else
  echo "실행 중인 프로세스 없음"
fi

# 2. 소스 교체 (temp에서 복사)
echo "[2/5] 소스 코드 교체"
cd $APP_DIR

# temp 디렉토리 확인
if [ ! -d "$TEMP_DIR" ] || [ -z "$(ls -A $TEMP_DIR)" ]; then
  echo "에러: temp 디렉토리가 비어있습니다. SCP 전송이 완료되지 않았습니다."
  exit 1
fi

# 기존 소스 삭제 (venv, logs는 유지)
rm -rf src app.py requirements.txt README.md data scripts test_all_task.py

# temp에서 복사
cp -r $TEMP_DIR/* $APP_DIR/
rm -rf $TEMP_DIR/*

echo "소스 코드 교체 완료"

# 3. 가상환경 및 의존성 설치
echo "[3/5] 의존성 설치"
cd $APP_DIR

# venv가 없을 때만 생성
if [ ! -d "venv" ]; then
  echo "가상환경 생성 중..."
  python3 -m venv venv
fi

. venv/bin/activate
pip install --no-cache-dir -r requirements.txt
echo "의존성 설치 완료"

# 4. 애플리케이션 시작
echo "[4/5] FastAPI 서버 실행"

# logs 디렉토리 생성
mkdir -p $APP_DIR/logs

cd $APP_DIR
setsid nohup $APP_DIR/venv/bin/python $APP_DIR/app.py > $APP_DIR/logs/ai.log 2>&1 < /dev/null &
APP_PID=$!
echo "서버 시작됨 (PID: $APP_PID)"

# 5. 시작 확인
echo "[5/5] 서버 시작 확인"

# 서버가 포트를 바인딩할 때까지 대기 (최대 15초)
for i in {1..15}; do
  if lsof -ti :$PORT > /dev/null 2>&1; then
    RUNNING_PID=$(lsof -ti :$PORT)
    echo "===== 배포 완료 ====="
    echo "AI 서버 실행 중 (port $PORT, PID: $RUNNING_PID)"
    exit 0
  fi
  echo "서버 시작 대기 중... ($i/15)"
  sleep 1
done

echo "===== 배포 실패 ====="
echo "서버가 시작되지 않았습니다."
echo ""
echo "최근 로그 (마지막 30줄):"
tail -30 $APP_DIR/logs/ai.log || echo "로그 파일을 찾을 수 없습니다"
exit 1
