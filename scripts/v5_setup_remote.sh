#!/usr/bin/env bash
# CipherChat v5 (INSECURE, e = 0) – build on a compute machine. Run from the synced checkout (scripts/v5_sync.sh).
# Needs the v3 toolchain first (scripts/v3_setup_remote.sh: NTL, NEXUS checkout with PhantomFHE, nlohmann/json).
#   1. copy PhantomFHE from the NEXUS checkout and patch it (cipherchat/v5/fhe/patch_phantom.py) -> $BUILD/phantom-src
#   2. build v5fhe (no bootstrapping) and, unless V5_BUILD_V3_E0=OFF, the v3 module with e = 0 -> $BUILD/v3e0/
# Env: BUILD (dir, default build_v5), TARGET (e.g. v5fhe), JOBS, V5_BUILD_V3_E0.
# The v3 build (build_v3, unpatched Phantom) is not touched.
set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
B=${BUILD:-build_v5}
source .venv/bin/activate
mkdir -p "$B"
python cipherchat/v5/fhe/patch_phantom.py ~/src/NEXUS/cuda/thirdparty/phantom-fhe "$B/phantom-src"
cmake -S cipherchat/v5/fhe -B "$B" -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(which python)" -DPHANTOM_V5_DIR="$ROOT/$B/phantom-src" \
  -DV5_BUILD_V3_E0="${V5_BUILD_V3_E0:-ON}" > "$B.cmake.log"
cmake --build "$B" -j"${JOBS:-20}" ${TARGET:+--target $TARGET} > "$B.make.log" 2>&1 || { tail -40 "$B.make.log"; exit 1; }
echo "built: $(ls $B/v5fhe*.so $B/v3e0/v3fhe*.so 2>/dev/null)"
