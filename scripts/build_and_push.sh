#!/bin/bash
# build_and_push.sh

set -e

# ============================================
# 설정
# ============================================
DOCKERHUB_USERNAME="jinsoo1218"
IMAGE_NAME="runpod_vllm"
VERSION="${1:-latest}"
PLATFORM="linux/amd64"

FULL_IMAGE_NAME="${DOCKERHUB_USERNAME}/${IMAGE_NAME}"

# ============================================
# 이미지 빌드 및 푸시
# ============================================
echo "============================================"
echo "Docker 이미지 빌드 및 푸시"
echo "============================================"
echo "이미지: ${FULL_IMAGE_NAME}:${VERSION}"
echo "플랫폼: ${PLATFORM}"
echo "============================================"
echo ""

echo "🔨 이미지 빌드 중..."
# 로그인이 안 되어 있으면 자동으로 에러가 발생하므로 확인 생략 가능
docker buildx build \
  --platform ${PLATFORM} \
  --tag ${FULL_IMAGE_NAME}:${VERSION} \
  --tag ${FULL_IMAGE_NAME}:latest \
  --push \
  --progress=plain \
  .

echo ""
echo "✅ 빌드 및 푸시 완료!"
echo ""
echo "📦 푸시된 이미지:"
echo "   - ${FULL_IMAGE_NAME}:${VERSION}"
echo "   - ${FULL_IMAGE_NAME}:latest"
echo ""
echo "🚀 이미지 실행 예제:"
echo "   docker run -p 8001:8001 ${FULL_IMAGE_NAME}:latest"