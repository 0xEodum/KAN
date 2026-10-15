cmake_minimum_required(VERSION 3.24)
project(KAN VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

option(KAN_ENABLE_CUDA "Build optional CUDA backends" OFF)
option(KAN_ALLOW_UNSUPPORTED_CUDA_COMPILER "Explicitly override nvcc host compiler version guard" OFF)
option(KAN_ENABLE_COVERAGE "Instrument CPU code with GCC gcov" OFF)
option(KAN_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(KAN_BUILD_EXAMPLES "Build standalone C++ examples" ON)
option(KAN_BUILD_PYTHON "Build optional pybind11/NumPy bindings" OFF)
option(KAN_BUILD_BENCHMARKS "Build frozen matched full-call benchmarks" OFF)
option(KAN_BENCH_RESIDENT "Include M2 resident backend in benchmarks" ON)
# Backlog C10: OFF is the parity build (CUDA kernels compiled with --fmad=false,
# the C2 behaviour); ON is the performance build (nvcc contracts a*b+c into FMA).
option(KAN_CUDA_FMA "Allow fused multiply-add contraction in the CUDA kernels (performance build)" OFF)

# No contraction into FMA in any in-tree host code (library, tests, examples,
# benchmarks, bindings, nvcc host compilation): CPU results must match the
# --fmad=false CUDA build and the guarded intermediates of the shared formulas,
# and tests compare against unfused host arithmetic. GCC defaults to
# -ffp-contract=fast in the GNU dialects and contracts wherever the target ISA
# has FMA (aarch64 always; x86-64 only with -march). Directory scope, so the
# flag is not exported to consumers of the installed targets.
if(NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
                        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-ffp-contract=off>)
endif()

# Backlog R4: sources are grouped by execution properties. core/ holds the
# carrier-independent Layer/Network/InputMap protocol, errors and shapes;
# carriers/ the family math (polynomial, trigonometric, local, rational; shared
# host/device formulas, validation, family operations); backends/cpu the CPU
# carrier loops; backends/cuda the optional kan::cuda target; init/ the
# initializers. Private headers are included relative to src/.
add_library(kan STATIC src/core/layer.cpp src/core/network.cpp src/core/input_map.cpp
            src/carriers/basis.cpp src/carriers/local/local_operations.cpp
            src/carriers/rational/rational.cpp src/carriers/rational/rational_operations.cpp
            src/backends/cpu/basis_edges.cpp src/backends/cpu/rational_edges.cpp
            src/init/initializers.cpp src/init/moments.cpp)
add_library(kan::kan ALIAS kan)
target_compile_features(kan PUBLIC cxx_std_20)
set_target_properties(kan PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kan PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(MSVC)
    target_compile_options(kan PRIVATE /W4 /permissive-)
else()
    target_compile_options(kan PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(KAN_ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR KAN_ENABLE_CUDA)
        message(FATAL_ERROR "Coverage requires a CPU-only GCC build")
    endif()
    target_compile_options(kan PRIVATE --coverage -O0 -g)
    target_link_options(kan PUBLIC --coverage)
endif()
if(KAN_ENABLE_ASAN)
    if(MSVC)
        target_compile_options(kan PUBLIC /fsanitize=address)
    else()
        target_compile_options(kan PUBLIC -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(kan PUBLIC -fsanitize=address)
    endif()
endif()

if(KAN_ENABLE_CUDA)
    if(KAN_ALLOW_UNSUPPORTED_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
        message(WARNING "nvcc host compiler guard explicitly overridden; validate this toolchain carefully")
    endif()
    # Default to the visible GPU (needs one at configure time); an explicit
    # -DCMAKE_CUDA_ARCHITECTURES or CUDAARCHS selects other targets.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    # The resident contraction engine uses the 64-bit cuBLAS API (CUDA 12+).
    find_package(CUDAToolkit 12 REQUIRED)
    add_library(kan_cuda STATIC src/backends/cuda/cuda_runtime.cpp src/backends/cuda/cuda.cu
                src/backends/cuda/resident.cu)
    add_library(kan::cuda ALIAS kan_cuda)
    target_include_directories(kan_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(kan_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON EXPORT_NAME cuda POSITION_INDEPENDENT_CODE ON)
    # Parity build (default): no FMA contraction in the kernels, preserving the
    # CPU's unfused intermediates and overflow checks (fusion can hide an
    # overflowing product through cancellation with a finite accumulator).
    # Performance build (KAN_CUDA_FMA=ON): nvcc's default --fmad=true.
    if(KAN_CUDA_FMA)
        target_compile_options(kan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
    else()
        target_compile_options(kan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()
    target_link_libraries(kan_cuda PUBLIC kan PRIVATE CUDA::cudart CUDA::cublas)
endif()

include(CTest)
if(BUILD_TESTING)
    # Backlog R5: tests are named by feature and grouped like src/ (core/, carriers/,
    # carriers/local/, carriers/rational/, init/, backends/cuda/; Python suites in
    # python/). Labels carry the execution requirement (cpu, cuda, python) and the
    # feature group (core, carriers, local, rational, init, resident), so for example
    # `ctest -L rational` runs the rational CPU, CUDA and Python suites. Executables
    # are built in the build root.
    # kan_add_test(<dir> <name> <labels> <timeout> <libraries>...) builds
    # tests/<dir>/<name>_test.cpp as <name>_test and registers it as CTest <name>.
    function(kan_add_test dir name labels timeout)
        add_executable(${name}_test tests/${dir}/${name}_test.cpp)
        target_include_directories(${name}_test PRIVATE tests)
        target_link_libraries(${name}_test PRIVATE ${ARGN})
        add_test(NAME ${name} COMMAND ${name}_test)
        set_tests_properties(${name} PROPERTIES LABELS "${labels}" TIMEOUT ${timeout})
    endfunction()
    foreach(suite layer network input_map residual residual_formula residual_training)
        kan_add_test(core ${suite} "cpu;core" 60 kan)
    endforeach()
    # Backlog M5: normalized Hermite functions (hermite_function).
    foreach(suite basis basis_config carrier hermite_function)
        kan_add_test(carriers ${suite} "cpu;carriers" 60 kan)
    endforeach()
    foreach(suite local_basis local_layer)
        kan_add_test(carriers/local ${suite} "cpu;local" 60 kan)
    endforeach()
    # Backlog C13: derivative-coefficient form of the spline input derivative (shared formulas).
    kan_add_test(carriers/local spline_derivative_formula "cpu;local" 60 kan)
    target_include_directories(spline_derivative_formula_test PRIVATE src)
    # Backlog M2: rational denominator policies (rational_policy).
    foreach(suite rational rational_layer rational_allocation rational_policy)
        kan_add_test(carriers/rational ${suite} "cpu;rational" 60 kan)
    endforeach()
    foreach(suite initializer initializer_training)
        kan_add_test(init ${suite} "cpu;init" 60 kan)
    endforeach()
    # Backlog M4: the deep-network training demonstration runs 3000 full-batch epochs per case.
    set_tests_properties(initializer_training PROPERTIES TIMEOUT 900)
    # Backlog M3: the residual formulas are internal (src/core); the demonstration
    # trains ten five-layer networks for 3000 epochs.
    target_include_directories(residual_formula_test PRIVATE src)
    set_tests_properties(residual_training PROPERTIES TIMEOUT 900)
    add_test(NAME package_custom_include COMMAND ${CMAKE_COMMAND}
             "-DKAN_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
             "-DKAN_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
             "-DKAN_CONFIG=$<CONFIG>"
             "-DKAN_COMPILER=${CMAKE_CXX_COMPILER}"
             "-DKAN_GENERATOR=${CMAKE_GENERATOR}"
             "-DKAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
             -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package_test.cmake")
    set_tests_properties(package_custom_include PROPERTIES LABELS "integration;cpu" TIMEOUT 120)
    add_test(NAME package_default_configuration COMMAND ${CMAKE_COMMAND}
             "-DKAN_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
             "-DKAN_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/default-config-regression"
             "-DKAN_CONFIG="
             "-DKAN_COMPILER=${CMAKE_CXX_COMPILER}"
             "-DKAN_GENERATOR=${CMAKE_GENERATOR}"
             "-DKAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
             -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package_test.cmake")
    set_tests_properties(package_default_configuration PROPERTIES LABELS "integration;cpu" TIMEOUT 120)
    if(KAN_ENABLE_CUDA)
        find_package(Threads REQUIRED)
        # Legacy M1 synchronous API (deprecated since backlog R7).
        kan_add_test(backends/cuda cuda "cuda" 120 kan::cuda CUDA::cudart Threads::Threads)
        # Backlog R7: kan::cuda::forward/backward are [[deprecated]], available() is not.
        # Probes compile with deprecation warnings as errors; the legacy one must fail.
        # R8: resident_only includes kan/resident.hpp alone and must see available().
        foreach(probe available resident_only legacy)
            add_executable(cuda_deprecation_${probe} EXCLUDE_FROM_ALL tests/backends/cuda/cuda_deprecation_probe.cpp)
            target_link_libraries(cuda_deprecation_${probe} PRIVATE kan::cuda)
            target_compile_options(cuda_deprecation_${probe} PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/we4996,-Werror=deprecated-declarations>)
            add_test(NAME cuda_deprecation_${probe} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target cuda_deprecation_${probe})
            set_tests_properties(cuda_deprecation_${probe} PROPERTIES LABELS "cuda;build" TIMEOUT 300 RESOURCE_LOCK kan_build_tree)
        endforeach()
        target_compile_definitions(cuda_deprecation_legacy PRIVATE KAN_PROBE_LEGACY)
        target_compile_definitions(cuda_deprecation_resident_only PRIVATE KAN_PROBE_RESIDENT_ONLY)
        set_tests_properties(cuda_deprecation_legacy PROPERTIES
            PASS_REGULAR_EXPRESSION "C4996.*ResidentNetwork;ResidentNetwork.*deprecated-declarations")
        # Resident executor, carrier-independent: review checks, C2 GEMM contraction
        # engine, C1 precision policy, R9 parameter upload, C9 graph-captured
        # training, C3 fused basis expansion (no stored derivative rows).
        kan_add_test(backends/cuda resident "cuda;resident" 120 kan::cuda Threads::Threads)
        foreach(suite resident_review resident_contraction resident_precision resident_upload)
            kan_add_test(backends/cuda ${suite} "cuda;resident" 120 kan::cuda)
        endforeach()
        kan_add_test(backends/cuda resident_training "cuda;resident" 180 kan::cuda)
        kan_add_test(backends/cuda resident_fused_expansion "cuda;resident" 120 kan::cuda CUDA::cudart)
        # Carrier and network features on the resident executor (parity with the CPU).
        kan_add_test(backends/cuda local_resident "cuda;local" 120 kan::cuda)
        # Backlog C5: sparse B-spline path; uses the private test hook (src/backends/cuda/resident_testing.hpp).
        kan_add_test(backends/cuda local_sparse_resident "cuda;local" 300 kan::cuda)
        target_include_directories(local_sparse_resident_test PRIVATE src)
        # Backlog C13: FP32/TF32 B-spline input VJP accuracy (same private hook).
        kan_add_test(backends/cuda local_spline_gradient_resident "cuda;local" 300 kan::cuda)
        target_include_directories(local_spline_gradient_resident_test PRIVATE src)
        # Backlog C12: trainable-RBF center/log-width reduction paths (same private hook).
        kan_add_test(backends/cuda local_rbf_reduction_resident "cuda;local" 300 kan::cuda)
        target_include_directories(local_rbf_reduction_resident_test PRIVATE src)
        # Backlog M2 (rational_policy_resident); C6-C8: rational kernel layout,
        # forward VJP checks, launch bounds (rational_resident_execution).
        foreach(suite rational_resident rational_policy_resident)
            kan_add_test(backends/cuda ${suite} "cuda;rational" 120 kan::cuda)
        endforeach()
        kan_add_test(backends/cuda rational_resident_execution "cuda;rational" 180 kan::cuda)
        kan_add_test(backends/cuda input_map_resident "cuda;core" 120 kan::cuda)
        # Backlog M3 phase 2: the residual branch on the resident executor.
        kan_add_test(backends/cuda residual_resident "cuda;core" 600 kan::cuda)
        # Backlog M4: initialized networks on the resident executor.
        kan_add_test(backends/cuda initializer_resident "cuda;init" 120 kan::cuda)
        # Explicit manual large-launch regression (trainable RBF); minimum-memory
        # requirement is outside the ordinary CTest hardware contract.
        add_executable(local_large_basis_test tests/backends/cuda/local_large_basis_test.cpp)
        target_link_libraries(local_large_basis_test PRIVATE kan::cuda CUDA::cudart)
    endif()
endif()

if(KAN_BUILD_EXAMPLES)
    add_executable(fit_polynomial examples/fit_polynomial.cpp)
    target_link_libraries(fit_polynomial PRIVATE kan)
    if(BUILD_TESTING)
        add_test(NAME fit_polynomial COMMAND fit_polynomial)
        set_tests_properties(fit_polynomial PROPERTIES LABELS "integration;cpu" TIMEOUT 60)
    endif()
endif()

if(KAN_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE KAN_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE KAN_PYBIND11_RESULT ERROR_QUIET)
    if(KAN_PYBIND11_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${KAN_PYBIND11_DIR}")
    endif()
    find_package(pybind11 3.0 CONFIG REQUIRED)
    pybind11_add_module(_kan python/bindings.cpp python/initializer_bindings.cpp)
    target_link_libraries(_kan PRIVATE kan)
    set_target_properties(_kan PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
    file(COPY python/kan DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python" PATTERN "__pycache__" EXCLUDE)
    add_custom_command(TARGET _kan POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${CMAKE_CURRENT_SOURCE_DIR}/python/kan" "$<TARGET_FILE_DIR:_kan>/kan")
    if(KAN_ENABLE_CUDA)
        target_compile_definitions(_kan PRIVATE KAN_PYTHON_CUDA)
        target_link_libraries(_kan PRIVATE kan::cuda)
        # cuBLAS is a runtime DLL on Windows; kan/__init__.py registers these
        # directories before importing _kan (Python ignores PATH there).
        set(KAN_DLL_PATHS_FILE "${CMAKE_CURRENT_BINARY_DIR}/kan_dll_paths.py")
        file(WRITE "${KAN_DLL_PATHS_FILE}"
             "# Generated by CMake: CUDA toolkit runtime DLL directories.
DIRECTORIES = (r\"${CUDAToolkit_BIN_DIR}/x64\", r\"${CUDAToolkit_BIN_DIR}\")
")
        add_custom_command(TARGET _kan POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                           "${KAN_DLL_PATHS_FILE}" "$<TARGET_FILE_DIR:_kan>/kan/_dll_paths.py")
    endif()
    set(KAN_PYTHON_INSTALL_DIR "python" CACHE STRING "Optional Python module install directory relative to prefix")
    install(TARGETS _kan LIBRARY DESTINATION ${KAN_PYTHON_INSTALL_DIR} RUNTIME DESTINATION ${KAN_PYTHON_INSTALL_DIR})
    install(DIRECTORY python/kan DESTINATION ${KAN_PYTHON_INSTALL_DIR}
            FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
    if(KAN_ENABLE_CUDA)
        install(FILES "${KAN_DLL_PATHS_FILE}" DESTINATION ${KAN_PYTHON_INSTALL_DIR}/kan RENAME _dll_paths.py)
    endif()
    if(BUILD_TESTING)
        set(KAN_PYTHON_TEST_ARGS "")
        if(KAN_ENABLE_CUDA)
            set(KAN_PYTHON_TEST_ARGS --cuda)
        endif()
        # Backlog R5: tests/python/<name>_test.py registered as CTest <name>, with the
        # feature group label of the corresponding C++ suites where there is one.
        function(kan_add_python_test name labels)
            add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E env
                     "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                     "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/${name}_test.py" ${KAN_PYTHON_TEST_ARGS})
            set_tests_properties(${name} PROPERTIES LABELS "${labels}" TIMEOUT 120)
        endfunction()
        kan_add_python_test(python "python;integration")
        kan_add_python_test(local_python "python;integration;local")
        kan_add_python_test(rational_python "python;integration;rational")
        kan_add_python_test(input_map_python "python;integration;core")
        kan_add_python_test(residual_python "python;integration;core")
        kan_add_python_test(initializer_python "python;integration;init")
        kan_add_python_test(hermite_function_python "python;integration;carriers")
        foreach(suite precision_python resident_upload_python training_python)
            kan_add_python_test(${suite} "python;integration;resident")
        endforeach()
    endif()
endif()

if(KAN_BUILD_BENCHMARKS)
    if(NOT KAN_ENABLE_CUDA)
        message(FATAL_ERROR "Matched benchmarks require KAN_ENABLE_CUDA")
    endif()
    # Backlog R5: frozen suites named by their case sets (formerly m2/m3/m4_benchmark):
    # basis (polynomial, Fourier, Gaussian RBF), local (B-spline, Mexican hat, trainable
    # RBF), rational. Cases, options and CSV columns are unchanged.
    add_executable(basis_benchmark benchmarks/basis_benchmark.cpp)
    target_link_libraries(basis_benchmark PRIVATE kan::cuda)
    add_executable(local_benchmark benchmarks/local_benchmark.cpp)
    target_link_libraries(local_benchmark PRIVATE kan::cuda)
    target_compile_definitions(local_benchmark PRIVATE KAN_BENCH_RESIDENT)
    add_executable(rational_benchmark benchmarks/rational_benchmark.cpp)
    target_link_libraries(rational_benchmark PRIVATE kan::cuda)
    target_compile_definitions(rational_benchmark PRIVATE KAN_BENCH_RESIDENT)
    if(KAN_BENCH_RESIDENT)
        target_compile_definitions(basis_benchmark PRIVATE KAN_BENCH_RESIDENT)
    endif()
endif()

include(CMakePackageConfigHelpers)
install(TARGETS kan EXPORT KANTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(KAN_ENABLE_CUDA)
    install(TARGETS kan_cuda EXPORT KANTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT KANTargets NAMESPACE kan:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KAN)
configure_package_config_file(cmake/KANConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/KANConfig.cmake
                             INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KAN)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/KANConfigVersion.cmake
                                COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KANConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/KANConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KAN)
