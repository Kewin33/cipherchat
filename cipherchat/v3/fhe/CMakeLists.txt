# CipherChat v3 – non-interactive CKKS engine on the GPU (PhantomFHE + NEXUS-CUDA bootstrapping).
#
# Third-party code is NOT vendored (NEXUS is GPL-3.0, PhantomFHE MIT); scripts/v3_setup_remote.sh fetches it:
#   NEXUS_CUDA_DIR  NEXUS-CUDA checkout (contains the modified PhantomFHE and the CKKS bootstrapper)
#   NTL_DIR         NTL install prefix (needed by the bootstrapper's Remez code)
#   JSON_INCLUDE    directory containing nlohmann/json.hpp
#
# Build (at-10-1-x, CUDA 12.0 has no sm_120 -> PTX compute_90, JIT-compiled for the RTX 5080):
#   cmake -S cipherchat/v3/fhe -B build_v3 -DCMAKE_CUDA_ARCHITECTURES=90 \
#         -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_CUDA_HOST_COMPILER=g++-11 && cmake --build build_v3 -j20
cmake_minimum_required(VERSION 3.20)
project(cipherchat_v3_fhe LANGUAGES CXX CUDA)

set(NEXUS_CUDA_DIR "$ENV{HOME}/src/NEXUS/cuda" CACHE PATH "NEXUS-CUDA checkout")
set(NTL_DIR "$ENV{HOME}/opt/ntl" CACHE PATH "NTL install prefix")
set(JSON_INCLUDE "$ENV{HOME}/src/include" CACHE PATH "dir with nlohmann/json.hpp")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --default-stream per-thread -Xcompiler -fPIC")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(PHANTOM_USE_CUDA_PTX)

set(PH ${NEXUS_CUDA_DIR}/thirdparty/phantom-fhe)
file(GLOB_RECURSE PHANTOM_SRC ${PH}/src/*.cu)
file(GLOB_RECURSE BOOT_SRC ${NEXUS_CUDA_DIR}/src/bootstrapping/*.cu)

add_library(nexusfhe STATIC ${PHANTOM_SRC} ${BOOT_SRC} ${NEXUS_CUDA_DIR}/src/ckks_evaluator.cu)
target_include_directories(nexusfhe PUBLIC
  ${PH}/include ${NEXUS_CUDA_DIR}/src ${NEXUS_CUDA_DIR}/src/bootstrapping
  ${NEXUS_CUDA_DIR}/src/bootstrapping/common ${NTL_DIR}/include ${JSON_INCLUDE})
target_link_libraries(nexusfhe PUBLIC ${NTL_DIR}/lib/libntl.a gmp m pthread)

add_executable(v3_bench_m1 src/bench_m1.cu)
target_include_directories(v3_bench_m1 PRIVATE src)
target_link_libraries(v3_bench_m1 PRIVATE nexusfhe)

# Python module v3fhe (pybind11 from the venv: -Dpybind11_DIR=$(python -m pybind11 --cmakedir))
find_package(Python 3.12 COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG)
if(pybind11_FOUND)
  pybind11_add_module(v3fhe src/pyv3fhe.cu)
  set_target_properties(v3fhe PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_include_directories(v3fhe PRIVATE src)
  target_link_libraries(v3fhe PRIVATE nexusfhe)
endif()
