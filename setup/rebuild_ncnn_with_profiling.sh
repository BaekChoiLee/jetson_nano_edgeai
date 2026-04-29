#!/bin/bash
# =============================================================================
# rebuild_ncnn_with_profiling.sh — -DNCNN_BENCHMARK=ON 으로 ncnn 재빌드
# =============================================================================
# 문제: layer_runtime 단계에서 "No per-layer timings in benchncnn output"
# 원인: ncnn이 -DNCNN_BENCHMARK=ON 없이 빌드됨
# 결과: benchncnn 바이너리가 per-layer timing 출력 안 함
# 해결: 별도 빌드 디렉토리에서 -DNCNN_BENCHMARK=ON 옵션으로 재빌드
# =============================================================================
set -e

NCNN_SRC="$HOME/ncnn"
BUILD_DIR="$HOME/ncnn/build_benchmark"
INSTALL_DIR="$HOME/ncnn_benchmark_build"

echo "=== ncnn 프로파일링 빌드 ==="
echo "소스: $NCNN_SRC"
echo "빌드: $BUILD_DIR"

if [ ! -d "$NCNN_SRC" ]; then
    echo "[ERROR] ncnn 소스 없음: $NCNN_SRC"
    echo "클론: git clone --recursive https://github.com/Tencent/ncnn.git ~/ncnn"
    exit 1
fi

# 기존 프로파일링 빌드가 있으면 재사용
if [ -d "$BUILD_DIR" ] && [ -f "$BUILD_DIR/tools/benchncnn" ]; then
    echo "기존 빌드 발견: $BUILD_DIR/tools/benchncnn"
    echo "재빌드를 강제하려면 $BUILD_DIR 삭제 후 재실행"
    BENCHNCNN_BIN="$BUILD_DIR/tools/benchncnn"
else
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake "$NCNN_SRC" \
        -DNCNN_BENCHMARK=ON \
        -DNCNN_VULKAN=ON \
        -DNCNN_BUILD_TOOLS=ON \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release

    # Jetson Nano는 CPU 2개 이하로 제한 (OOM 방지)
    make -j2 benchncnn

    BENCHNCNN_BIN="$BUILD_DIR/tools/benchncnn"
    echo "빌드 완료: $BENCHNCNN_BIN"
fi

# 확인
echo "=== 빌드 확인 ==="
"$BENCHNCNN_BIN" --help 2>&1 | head -5 || true
"$BENCHNCNN_BIN" 1 1 2>/dev/null | head -3 || true

echo ""
echo "=== 환경 변수 설정 (rebenchmark 실행 전 적용) ==="
echo "export NCNN_BENCHNCNN_BIN='$BENCHNCNN_BIN'"
echo ""
echo "또는 rebenchmark_clean_v2.sh 실행 시:"
echo "  NCNN_BENCHNCNN_BIN='$BENCHNCNN_BIN' bash rebenchmark_clean_v2.sh ..."
