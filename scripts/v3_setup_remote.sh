#!/usr/bin/env bash
# One-time setup of the v3 FHE toolchain on a compute machine (no sudo needed; everything under $HOME).
#   - NTL 11.5.1 -> ~/opt/ntl (needed by the NEXUS bootstrapper's Remez code)
#   - NEXUS (GPL-3.0) pinned commit -> ~/src/NEXUS (its cuda/ dir has a modified PhantomFHE + CKKS bootstrapping)
#   - nlohmann/json 3.11.3 header -> ~/src/include
# Then builds cipherchat/v3/fhe into ~/cipherchat/build_v3 (CUDA 12.0: PTX compute_90, JIT on the RTX 5080).
set -e
NEXUS_COMMIT=e15058c7f8f58fcf037c4709211109cb1be7d1e0
mkdir -p ~/src ~/opt ~/src/include/nlohmann
if [ ! -f ~/opt/ntl/lib/libntl.a ]; then
  cd ~/src && mkdir -p ntl_build && cd ntl_build
  [ -f ntl-11.5.1.tar.gz ] || curl -sSLO https://libntl.org/ntl-11.5.1.tar.gz
  tar xzf ntl-11.5.1.tar.gz && cd ntl-11.5.1/src
  ./configure PREFIX=$HOME/opt/ntl GMP_PREFIX=/usr NTL_THREADS=on SHARED=off NATIVE=off CXXFLAGS="-O2 -fPIC"
  make -j16 && make install
fi
if [ ! -d ~/src/NEXUS ]; then
  git clone -q https://github.com/zju-abclab/NEXUS.git ~/src/NEXUS
fi
(cd ~/src/NEXUS && git fetch -q --depth 1 origin $NEXUS_COMMIT 2>/dev/null || true; git checkout -q $NEXUS_COMMIT 2>/dev/null || true)
[ -f ~/src/include/nlohmann/json.hpp ] || curl -sSL -o ~/src/include/nlohmann/json.hpp \
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
cd ~/cipherchat
source ~/cipherchat/.venv/bin/activate
cmake -S cipherchat/v3/fhe -B build_v3 -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(which python)" > build_v3.cmake.log
cmake --build build_v3 -j20 > build_v3.make.log 2>&1 || { tail -40 build_v3.make.log; exit 1; }
echo "built: $(ls build_v3/v3_* build_v3/v3fhe*.so)"
