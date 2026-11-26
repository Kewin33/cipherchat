# CipherChat v5 – INSECURE upper-bound experiments: CKKS with RLWE error e = 0 (PhantomFHE, patched copy).
#
#   *** NOT SECURE. The secret key follows from the public key. Only for measuring what bootstrapping and the
#   *** security parameters cost (docs/V5.md).
#
# Phantom is NOT vendored: scripts/v5_setup_remote.sh copies it from the NEXUS checkout and applies
# cipherchat/v5/fhe/patch_phantom.py into ${PHANTOM_V5_DIR}. Every patch is behind a macro; the targets set them:
#
#   v5fhe   (Python)  PHANTOM_ZERO_ERROR + PHANTOM_V5_LONG_CHAIN + PHANTOM_SINGLE_DIGIT_KS + PHANTOM_V5_CENTERED,
#                     no bootstrapper
#   v3fhe   (Python, option V5_BUILD_V3_E0, written to <build>/v3e0/)  the unchanged v3 module
#           (cipherchat/v3/fhe/src/pyv3fhe.cu incl. NEXUS bootstrapping) linked against Phantom with ONLY
#           PHANTOM_ZERO_ERROR -> v3 parameters with e = 0, isolates the effect of the error term itself
#
# Build (at-10-1-x, CUDA 12.0 -> PTX compute_90):
#   cmake -S cipherchat/v5/fhe -B build_v5 -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CXX_COMPILER=g++-11 \
#         -DCMAKE_CUDA_HOST_COMPILER=g++-11 -DPHANTOM_V5_DIR=$PWD/build_v5/phantom-src && cmake --build build_v5 -j20
cmake_minimum_required(VERSION 3.20)
project(cipherchat_v5_fhe LANGUAGES CXX CUDA)

set(PHANTOM_V5_DIR "${CMAKE_BINARY_DIR}/phantom-src" CACHE PATH "patched PhantomFHE copy (patch_phantom.py)")
set(NEXUS_CUDA_DIR "$ENV{HOME}/src/NEXUS/cuda" CACHE PATH "NEXUS-CUDA checkout (only for V5_BUILD_V3_E0)")
set(NTL_DIR "$ENV{HOME}/opt/ntl" CACHE PATH "NTL install prefix (only for V5_BUILD_V3_E0)")
set(JSON_INCLUDE "$ENV{HOME}/src/include" CACHE PATH "dir with nlohmann/json.hpp")
option(V5_BUILD_V3_E0 "also build the v3 module (with bootstrapping) against Phantom with e = 0" ON)

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

if(NOT EXISTS ${PHANTOM_V5_DIR}/src/prng.cu)
  message(FATAL_ERROR "patched Phantom not found in ${PHANTOM_V5_DIR}; run cipherchat/v5/fhe/patch_phantom.py")
endif()
file(GLOB_RECURSE PHANTOM_SRC ${PHANTOM_V5_DIR}/src/*.cu)

# ---- v5: e = 0, long chain, single-digit key switching, no bootstrapping
add_library(phantom_v5 STATIC ${PHANTOM_SRC})
target_include_directories(phantom_v5 PUBLIC ${PHANTOM_V5_DIR}/include ${JSON_INCLUDE})
target_compile_definitions(phantom_v5 PUBLIC PHANTOM_ZERO_ERROR PHANTOM_V5_LONG_CHAIN PHANTOM_SINGLE_DIGIT_KS
  PHANTOM_V5_CENTERED)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(v5fhe src/pyv5fhe.cu)
set_target_properties(v5fhe PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(v5fhe PRIVATE phantom_v5)

# ---- v3 module with e = 0 (same parameters and bootstrapping as v3, only the error term removed)
if(V5_BUILD_V3_E0)
  file(GLOB_RECURSE BOOT_SRC ${NEXUS_CUDA_DIR}/src/bootstrapping/*.cu)
  add_library(nexus_e0 STATIC ${PHANTOM_SRC} ${BOOT_SRC} ${NEXUS_CUDA_DIR}/src/ckks_evaluator.cu)
  target_include_directories(nexus_e0 PUBLIC
    ${PHANTOM_V5_DIR}/include ${NEXUS_CUDA_DIR}/src ${NEXUS_CUDA_DIR}/src/bootstrapping
    ${NEXUS_CUDA_DIR}/src/bootstrapping/common ${NTL_DIR}/include ${JSON_INCLUDE})
  target_compile_definitions(nexus_e0 PUBLIC PHANTOM_ZERO_ERROR)
  target_link_libraries(nexus_e0 PUBLIC ${NTL_DIR}/lib/libntl.a gmp m pthread)
  pybind11_add_module(v3fhe_e0 ${CMAKE_CURRENT_SOURCE_DIR}/../../v3/fhe/src/pyv3fhe.cu)
  set_target_properties(v3fhe_e0 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON OUTPUT_NAME v3fhe
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/v3e0)
  target_include_directories(v3fhe_e0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../v3/fhe/src)
  target_link_libraries(v3fhe_e0 PRIVATE nexus_e0)
endif()
