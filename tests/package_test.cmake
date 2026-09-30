# Real installed consumer: exercise a GNUInstallDirs override, not the build tree.
foreach(required KAN_SOURCE_DIR KAN_BUILD_DIR KAN_CONFIG KAN_COMPILER KAN_GENERATOR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required}")
    endif()
endforeach()
function(checked_command)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Command failed (${result}): ${ARGN}\n${output}\n${error}")
    endif()
endfunction()
set(regression_root "${KAN_BUILD_DIR}/package-regression")
set(tool_args -G "${KAN_GENERATOR}" "-DCMAKE_CXX_COMPILER=${KAN_COMPILER}")
if(KAN_MAKE_PROGRAM)
    list(APPEND tool_args "-DCMAKE_MAKE_PROGRAM=${KAN_MAKE_PROGRAM}")
endif()
checked_command("${CMAKE_COMMAND}" -S "${KAN_SOURCE_DIR}" -B "${regression_root}/build"
                ${tool_args} "-DCMAKE_BUILD_TYPE=${KAN_CONFIG}"
                -DKAN_ENABLE_CUDA=OFF -DBUILD_TESTING=OFF -DKAN_BUILD_EXAMPLES=OFF
                -DCMAKE_INSTALL_INCLUDEDIR=kan-headers)
checked_command("${CMAKE_COMMAND}" --build "${regression_root}/build" --config "${KAN_CONFIG}" --parallel)
checked_command("${CMAKE_COMMAND}" --install "${regression_root}/build" --config "${KAN_CONFIG}" --prefix "${regression_root}/install")
checked_command("${CMAKE_COMMAND}" -S "${KAN_SOURCE_DIR}/tests/consumer" -B "${regression_root}/consumer"
                ${tool_args} "-DCMAKE_BUILD_TYPE=${KAN_CONFIG}" "-DCMAKE_PREFIX_PATH=${regression_root}/install")
checked_command("${CMAKE_COMMAND}" --build "${regression_root}/consumer" --config "${KAN_CONFIG}" --parallel)
checked_command("${CMAKE_CTEST_COMMAND}" --test-dir "${regression_root}/consumer" -C "${KAN_CONFIG}" --output-on-failure)
message(STATUS "Custom include directory installed-package consumer passed")
