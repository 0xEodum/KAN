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

add_library(kan STATIC src/basis.cpp src/rational.cpp src/layer.cpp src/network.cpp src/families.cpp src/input_map.cpp
            src/carriers/basis_edges.cpp src/carriers/rational_edges.cpp)
add_library(kan::kan ALIAS kan)
target_compile_features(kan PUBLIC cxx_std_20)
set_target_properties(kan PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kan PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(MSVC)
    target_compile_options(kan PRIVATE /W4 /permissive-)
else()
    # No contraction into FMA: CPU results must match the --fmad=false CUDA
    # build and the guarded intermediates of the shared formulas.
    target_compile_options(kan PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
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
    enable_language(CUDA)
    # The resident contraction engine uses the 64-bit cuBLAS API (CUDA 12+).
    find_package(CUDAToolkit 12 REQUIRED)
    add_library(kan_cuda STATIC src/cuda_runtime.cpp src/cuda.cu src/resident.cu)
    add_library(kan::cuda ALIAS kan_cuda)
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
    foreach(suite basis basis_config carrier layer network input_map m3_basis m3_layer m4_rational m4_layer m4_allocation)
        add_executable(${suite}_test tests/${suite}_test.cpp)
        target_include_directories(${suite}_test PRIVATE tests)
        target_link_libraries(${suite}_test PRIVATE kan)
        add_test(NAME ${suite} COMMAND ${suite}_test)
        set_tests_properties(${suite} PROPERTIES LABELS "cpu" TIMEOUT 60)
    endforeach()
    # Backlog M2: rational denominator policies.
    add_executable(rational_policy_test tests/rational_policy_test.cpp)
    target_include_directories(rational_policy_test PRIVATE tests)
    target_link_libraries(rational_policy_test PRIVATE kan)
    add_test(NAME rational_policy COMMAND rational_policy_test)
    set_tests_properties(rational_policy PROPERTIES LABELS "cpu" TIMEOUT 60)
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
        add_executable(cuda_test tests/cuda_test.cpp)
        target_include_directories(cuda_test PRIVATE tests)
        find_package(Threads REQUIRED)
        target_link_libraries(cuda_test PRIVATE kan::cuda CUDA::cudart Threads::Threads)
        add_test(NAME cuda COMMAND cuda_test)
        set_tests_properties(cuda PROPERTIES LABELS "cuda" TIMEOUT 120)
        # Backlog R7: kan::cuda::forward/backward are [[deprecated]], available() is not.
        # Probes compile with deprecation warnings as errors; the legacy one must fail.
        # R8: resident_only includes kan/resident.hpp alone and must see available().
        foreach(probe available resident_only legacy)
            add_executable(cuda_deprecation_${probe} EXCLUDE_FROM_ALL tests/cuda_deprecation_probe.cpp)
            target_link_libraries(cuda_deprecation_${probe} PRIVATE kan::cuda)
            target_compile_options(cuda_deprecation_${probe} PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/we4996,-Werror=deprecated-declarations>)
            add_test(NAME cuda_deprecation_${probe} COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target cuda_deprecation_${probe})
            set_tests_properties(cuda_deprecation_${probe} PROPERTIES LABELS "cuda;build" TIMEOUT 300 RESOURCE_LOCK kan_build_tree)
        endforeach()
        target_compile_definitions(cuda_deprecation_legacy PRIVATE KAN_PROBE_LEGACY)
        target_compile_definitions(cuda_deprecation_resident_only PRIVATE KAN_PROBE_RESIDENT_ONLY)
        set_tests_properties(cuda_deprecation_legacy PROPERTIES
            PASS_REGULAR_EXPRESSION "C4996.*ResidentNetwork;ResidentNetwork.*deprecated-declarations")
        add_executable(resident_test tests/resident_test.cpp)
        target_include_directories(resident_test PRIVATE tests)
        target_link_libraries(resident_test PRIVATE kan::cuda Threads::Threads)
        add_test(NAME resident COMMAND resident_test)
        set_tests_properties(resident PROPERTIES LABELS "cuda" TIMEOUT 120)
        add_executable(resident_review_test tests/resident_review_test.cpp)
        target_include_directories(resident_review_test PRIVATE tests)
        target_link_libraries(resident_review_test PRIVATE kan::cuda)
        add_test(NAME resident_review COMMAND resident_review_test)
        set_tests_properties(resident_review PROPERTIES LABELS "cuda" TIMEOUT 120)
        add_executable(m3_resident_test tests/m3_resident_test.cpp)
        target_include_directories(m3_resident_test PRIVATE tests)
        target_link_libraries(m3_resident_test PRIVATE kan::cuda)
        add_test(NAME m3_resident COMMAND m3_resident_test)
        set_tests_properties(m3_resident PROPERTIES LABELS "cuda" TIMEOUT 120)
        add_executable(input_map_resident_test tests/input_map_resident_test.cpp)
        target_include_directories(input_map_resident_test PRIVATE tests)
        target_link_libraries(input_map_resident_test PRIVATE kan::cuda)
        add_test(NAME input_map_resident COMMAND input_map_resident_test)
        set_tests_properties(input_map_resident PROPERTIES LABELS "cuda" TIMEOUT 120)
        add_executable(m4_resident_test tests/m4_resident_test.cpp)
        target_include_directories(m4_resident_test PRIVATE tests)
        target_link_libraries(m4_resident_test PRIVATE kan::cuda)
        add_test(NAME m4_resident COMMAND m4_resident_test)
        set_tests_properties(m4_resident PROPERTIES LABELS "cuda" TIMEOUT 120)
        add_executable(rational_policy_resident_test tests/rational_policy_resident_test.cpp)
        target_include_directories(rational_policy_resident_test PRIVATE tests)
        target_link_libraries(rational_policy_resident_test PRIVATE kan::cuda)
        add_test(NAME rational_policy_resident COMMAND rational_policy_resident_test)
        set_tests_properties(rational_policy_resident PROPERTIES LABELS "cuda" TIMEOUT 120)
        # Backlog C2: GEMM contraction engine.
        add_executable(resident_contraction_test tests/resident_contraction_test.cpp)
        target_include_directories(resident_contraction_test PRIVATE tests)
        target_link_libraries(resident_contraction_test PRIVATE kan::cuda)
        add_test(NAME resident_contraction COMMAND resident_contraction_test)
        set_tests_properties(resident_contraction PROPERTIES LABELS "cuda" TIMEOUT 120)
        # Backlog C1: resident precision policy (FP32 executor).
        add_executable(resident_precision_test tests/resident_precision_test.cpp)
        target_include_directories(resident_precision_test PRIVATE tests)
        target_link_libraries(resident_precision_test PRIVATE kan::cuda)
        add_test(NAME resident_precision COMMAND resident_precision_test)
        set_tests_properties(resident_precision PROPERTIES LABELS "cuda" TIMEOUT 120)
        # Backlog R9: parameter upload into an existing executor.
        add_executable(resident_upload_test tests/resident_upload_test.cpp)
        target_include_directories(resident_upload_test PRIVATE tests)
        target_link_libraries(resident_upload_test PRIVATE kan::cuda)
        add_test(NAME resident_upload COMMAND resident_upload_test)
        set_tests_properties(resident_upload PROPERTIES LABELS "cuda" TIMEOUT 120)
        # Backlog C9: graph-captured training steps, deferred status, on-device MSE.
        add_executable(resident_training_test tests/resident_training_test.cpp)
        target_include_directories(resident_training_test PRIVATE tests)
        target_link_libraries(resident_training_test PRIVATE kan::cuda)
        add_test(NAME resident_training COMMAND resident_training_test)
        set_tests_properties(resident_training PROPERTIES LABELS "cuda" TIMEOUT 180)
        # Explicit manual large-launch regression; minimum-memory requirement is
        # outside the ordinary CTest hardware contract.
        add_executable(m3_large_basis_test tests/m3_large_basis_test.cpp)
        target_link_libraries(m3_large_basis_test PRIVATE kan::cuda CUDA::cudart)
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
    pybind11_add_module(_kan python/bindings.cpp)
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
        add_test(NAME python COMMAND ${CMAKE_COMMAND} -E env
                 "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                 "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_test.py" ${KAN_PYTHON_TEST_ARGS})
        set_tests_properties(python PROPERTIES LABELS "python;integration" TIMEOUT 120)
        add_test(NAME m3_python COMMAND ${CMAKE_COMMAND} -E env
                 "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                 "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/m3_python_test.py" ${KAN_PYTHON_TEST_ARGS})
        set_tests_properties(m3_python PROPERTIES LABELS "python;integration" TIMEOUT 120)
        add_test(NAME m4_python COMMAND ${CMAKE_COMMAND} -E env
                 "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                 "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/m4_python_test.py" ${KAN_PYTHON_TEST_ARGS})
        set_tests_properties(m4_python PROPERTIES LABELS "python;integration" TIMEOUT 120)
        add_test(NAME input_map_python COMMAND ${CMAKE_COMMAND} -E env
                 "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                 "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/input_map_python_test.py" ${KAN_PYTHON_TEST_ARGS})
        set_tests_properties(input_map_python PROPERTIES LABELS "python;integration" TIMEOUT 120)
        add_test(NAME precision_python COMMAND ${CMAKE_COMMAND} -E env
                 "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                 "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/precision_python_test.py" ${KAN_PYTHON_TEST_ARGS})
        set_tests_properties(precision_python PROPERTIES LABELS "python;integration" TIMEOUT 120)
        add_test(NAME resident_upload_python COMMAND ${CMAKE_COMMAND} -E env
                 "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                 "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/resident_upload_python_test.py" ${KAN_PYTHON_TEST_ARGS})
        set_tests_properties(resident_upload_python PROPERTIES LABELS "python;integration" TIMEOUT 120)
        add_test(NAME training_python COMMAND ${CMAKE_COMMAND} -E env
                 "PYTHONPATH=$<TARGET_FILE_DIR:_kan>"
                 "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/training_python_test.py" ${KAN_PYTHON_TEST_ARGS})
        set_tests_properties(training_python PROPERTIES LABELS "python;integration" TIMEOUT 120)
    endif()
endif()

if(KAN_BUILD_BENCHMARKS)
    if(NOT KAN_ENABLE_CUDA)
        message(FATAL_ERROR "Matched benchmarks require KAN_ENABLE_CUDA")
    endif()
    add_executable(m2_benchmark benchmarks/m2_benchmark.cpp)
    target_link_libraries(m2_benchmark PRIVATE kan::cuda)
    add_executable(m3_benchmark benchmarks/m3_benchmark.cpp)
    target_link_libraries(m3_benchmark PRIVATE kan::cuda)
    target_compile_definitions(m3_benchmark PRIVATE KAN_BENCH_RESIDENT)
    add_executable(m4_benchmark benchmarks/m4_benchmark.cpp)
    target_link_libraries(m4_benchmark PRIVATE kan::cuda)
    target_compile_definitions(m4_benchmark PRIVATE KAN_BENCH_RESIDENT)
    if(KAN_BENCH_RESIDENT)
        target_compile_definitions(m2_benchmark PRIVATE KAN_BENCH_RESIDENT)
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
