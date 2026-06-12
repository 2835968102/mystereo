#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_NAME="${IMAGE_NAME:-mycalib:docker}"
BASE_IMAGE="${BASE_IMAGE:-ubuntu:22.04}"

docker build \
  -f packaging/Dockerfile \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$IMAGE_NAME" \
  .

echo
echo "Built Docker image: $IMAGE_NAME"
