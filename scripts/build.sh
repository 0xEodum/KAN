#!/usr/bin/env bash
# Linux equivalent of scripts/build.ps1: configure, build and optionally test.
# Usage (any directory):
#   scripts/build.sh [--cuda] [--cuda-fma] [--python] [--benchmarks] [--asan] [--test]
#                    [--config Release|Debug|RelWithDebInfo|MinSizeRel] [--build-dir DIR] [--python-exe PATH]
#                    [--cuda-arch ARCHS] [--allow-unsupported-cuda-compiler]
# Defaults: Release; build directory build-cuda / build-cuda-fma / build-asan / build-cpu;
# CUDA architectures from CMake (the visible GPU unless --cuda-arch or CUDAARCHS is given);
# Python from .venv/bin/python.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda=OFF fma=OFF python=OFF bench=OFF asan=OFF override=OFF test=0
config=Release build_dir="" python_exe="" cuda_arch=""
while (($#)); do
    case $1 in
        --cuda) cuda=ON ;;
        --cuda-fma) cuda=ON fma=ON ;;
        --python) python=ON ;;
        --benchmarks) bench=ON ;;
        --asan) asan=ON ;;
        --test) test=1 ;;
        --allow-unsupported-cuda-compiler) override=ON ;;
        --config) config=$2; shift ;;
        --build-dir) build_dir=$2; shift ;;
        --python-exe) python_exe=$2; shift ;;
        --cuda-arch) cuda_arch=$2; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
case $config in Release|Debug|RelWithDebInfo|MinSizeRel) ;; *) echo "invalid --config: $config" >&2; exit 2 ;; esac
if [[ -z $build_dir ]]; then
    if [[ $fma == ON ]]; then build_dir=build-cuda-fma
    elif [[ $cuda == ON ]]; then build_dir=build-cuda
    elif [[ $asan == ON ]]; then build_dir=build-asan
    else build_dir=build-cpu; fi
fi
[[ $build_dir = /* ]] || build_dir="$root/$build_dir"
args=(-S "$root" -B "$build_dir" "-DCMAKE_BUILD_TYPE=$config" "-DKAN_ENABLE_CUDA=$cuda" "-DKAN_CUDA_FMA=$fma"
      "-DKAN_ENABLE_ASAN=$asan" "-DKAN_ALLOW_UNSUPPORTED_CUDA_COMPILER=$override" "-DKAN_BUILD_PYTHON=$python"
      "-DKAN_BUILD_BENCHMARKS=$bench")
[[ -n $cuda_arch ]] && args+=("-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch")
if [[ $python == ON ]]; then
    args+=("-DPython_EXECUTABLE=${python_exe:-$root/.venv/bin/python}")
fi
cmake "${args[@]}"
cmake --build "$build_dir" --parallel "$(nproc)"
if ((test)); then
    ctest --test-dir "$build_dir" --output-on-failure
fi
