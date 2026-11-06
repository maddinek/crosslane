# Included by the HIP top-level injection after strata-dsv41 exists. The host compiler and flags
# are the reference's CXX ones, never icpx; both model DSOs consume this exact shared library.
if(NOT TARGET strata-row2-cpu)
  if(NOT STRATA_NATIVE_EXPERTS OR NOT TARGET strata_kernels_cpu)
    message(FATAL_ERROR "row2 CPU lane needs the reference STRATA_NATIVE_EXPERTS=ON/strata_kernels_cpu")
  endif()
  add_library(strata-row2-cpu SHARED
    "${W}/../../moe/cpu_lane.cpp"
    "${CMAKE_SOURCE_DIR}/src/dsv41/ds4_gg_cpu.cpp"
    "${CMAKE_SOURCE_DIR}/src/dsv41/ds4_iq_cpu.cpp"
    "${CMAKE_SOURCE_DIR}/src/dsv41/ds4_gguf_cpu.cpp"
    "${CMAKE_SOURCE_DIR}/src/dsv41/ds4_tq_cpu.cpp"
    "${CMAKE_SOURCE_DIR}/src/dsv41/vendor/exl3_cpu.cpp")
  if(STRATA_FORK_WARM)
    target_sources(strata-row2-cpu PRIVATE "${SYCLROOT}/../warm/runtime.cpp")
    target_compile_definitions(strata-row2-cpu PRIVATE STRATA_FORK_WARM=1)
    add_executable(strata-cpu-lane-bench "${SYCLROOT}/../warm/cpu_lane_bench.cpp" "${W}/pack_source.cpp")
    target_include_directories(strata-cpu-lane-bench PRIVATE "${CMAKE_SOURCE_DIR}/src/dsv41")
    target_compile_features(strata-cpu-lane-bench PRIVATE cxx_std_20)
    target_compile_definitions(strata-cpu-lane-bench PRIVATE STRATA_FORK_WARM=1)
    target_link_libraries(strata-cpu-lane-bench PRIVATE strata-row2-cpu)
    set_target_properties(strata-cpu-lane-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      BUILD_RPATH_USE_ORIGIN YES BUILD_WITH_INSTALL_RPATH YES INSTALL_RPATH "$ORIGIN")
  endif()
  # Copy only host-applicable properties; GPU runtime forced headers are unnecessary here.
  get_target_property(row2_includes strata-dsv41 INCLUDE_DIRECTORIES)
  target_include_directories(strata-row2-cpu PRIVATE ${row2_includes} "${CMAKE_SOURCE_DIR}/src/dsv41")
  target_compile_definitions(strata-row2-cpu PRIVATE STRATA_DSV41_TQ_CPU=1)
  # No -march/native/fast-math override: same CMAKE_CXX_FLAGS[_RELEASE] as upstream host rows.
  target_link_libraries(strata-row2-cpu PRIVATE strata_kernels_cpu Threads::Threads)
  set_target_properties(strata-row2-cpu PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    POSITION_INDEPENDENT_CODE ON BUILD_RPATH_USE_ORIGIN YES BUILD_WITH_INSTALL_RPATH YES
    INSTALL_RPATH "$ORIGIN")
  # Rebuild CPU dependency archives as PIC in this fresh configure, never the accepted build tree.
  foreach(dep strata_kernels_cpu ggml ggml-base ggml-cpu)
    if(TARGET ${dep})
      set_property(TARGET ${dep} PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()
  endforeach()
  add_executable(strata-row2-cpu-fixture "${SYCLROOT}/../fixtures/moe/cpu_lane_fixture.cpp")
  target_include_directories(strata-row2-cpu-fixture PRIVATE "${CMAKE_SOURCE_DIR}/src/dsv41")
  target_compile_features(strata-row2-cpu-fixture PRIVATE cxx_std_20)
  set_target_properties(strata-row2-cpu-fixture PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_link_libraries(strata-row2-cpu-fixture PRIVATE strata-row2-cpu)
  add_executable(strata-row2-pool-fixture "${SYCLROOT}/../fixtures/moe/pool_fixture.cpp")
  target_compile_features(strata-row2-pool-fixture PRIVATE cxx_std_20)
  set_target_properties(strata-row2-pool-fixture PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_link_libraries(strata-row2-pool-fixture PRIVATE Threads::Threads)
endif()
