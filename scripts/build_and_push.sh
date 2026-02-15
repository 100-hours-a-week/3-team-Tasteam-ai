#!/bin/bash
# API 이미지 빌드 후 Docker Hub 푸시
# 사용: ./scripts/build_and_push.sh [VERSION] [cuda|cpu]
# 예:   ./scripts/build_and_push.sh 1.0.0 cuda
#      ./scripts/build_and_push.sh latest cpu

set -e

# ============================================
# 설정 (환경 변수 또는 기본값)
# ============================================
# docker-compose는 DOCKERHUB_USER 사용 → 스크립트도 DOCKERHUB_USER fallback
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-$DOCKERHUB_USER}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-tasteam_pipeline}"
VERSION="${1:-latest}"
VARIANT="${2:-cuda}"   # cuda | cpu
PLATFORM="${PLATFORM:-linux/amd64}"

if [ -z "$DOCKERHUB_USERNAME" ]; then
  echo "❌ DOCKERHUB_USERNAME이 비어 있습니다."
  echo "   예: DOCKERHUB_USERNAME=myuser ./scripts/build_and_push.sh 1.0.0 cuda"
  echo "   또는 .env에 DOCKERHUB_USERNAME=myuser 설정 후: source .env 2>/dev/null; ./scripts/build_and_push.sh"
  echo "   (docker-compose와 맞추려면 .env에 DOCKERHUB_USER=myuser, IMAGE_TAG=1.0.0 도 설정)"
  exit 1
fi

case "$VARIANT" in
  cpu)
    DOCKERFILE="dockerfile"
    TAG_SUFFIX="-cpu"
    ;;
  cuda)
    DOCKERFILE="Dockerfile.runpod-vllm"
    TAG_SUFFIX=""
    ;;
  *)
    echo "❌ VARIANT는 cuda 또는 cpu 여야 합니다. (입력: $VARIANT)"
    exit 1
    ;;
esac

FULL_IMAGE_NAME="${DOCKERHUB_USERNAME}/${IMAGE_NAME}"
TAG_VERSION="${FULL_IMAGE_NAME}:${VERSION}${TAG_SUFFIX}"
TAG_LATEST="${FULL_IMAGE_NAME}:latest${TAG_SUFFIX}"

# ============================================
# 이미지 빌드 및 푸시
# ============================================
echo "============================================"
echo "Docker 이미지 빌드 및 푸시 (Docker Hub)"
echo "============================================"
echo "  레지스트리: Docker Hub"
echo "  이미지:     ${FULL_IMAGE_NAME}"
echo "  Dockerfile: ${DOCKERFILE} (${VARIANT})"
echo "  태그:       ${TAG_VERSION}, ${TAG_LATEST}"
echo "  플랫폼:     ${PLATFORM}"
echo "============================================"
echo ""

echo "🔨 이미지 빌드 중... (docker buildx build)"
docker buildx build \
  --platform "${PLATFORM}" \
  --file "${DOCKERFILE}" \
  --tag "${TAG_VERSION}" \
  --tag "${TAG_LATEST}" \
  --push \
  --progress=plain \
  .

echo ""
echo "✅ 빌드 및 푸시 완료!"
echo ""
echo "📦 푸시된 이미지:"
echo "   - ${TAG_VERSION}"
echo "   - ${TAG_LATEST}"
echo ""
echo "🚀 이미지 실행 예제:"
if [ "$VARIANT" = "cuda" ]; then
  echo "   docker run --gpus all -p 8001:8001 ${TAG_LATEST}"
else
  echo "   docker run -p 8001:8001 ${TAG_LATEST}"
fi
echo ""
