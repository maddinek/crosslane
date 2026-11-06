# HIP stage targets, included (deferred) at the end of upstream Strata's top-level CMakeLists.txt via
# hip_top_level.cmake. Upstream is configured exactly as the accepted private build
# (results/strata-cross-vendor/hip-build-a1/configure-argv.txt); this fragment only ADDS targets:
#   libstrata-stage-hip-model.so  (v1 transport + v2 backend, hip_backend.cpp)
#   strata-dsv41-ref-t1           (AMD-only T=1 reference over upstream Model)
# Upstream sources are compiled with strata-dsv41's own sources, include dirs, definitions, options,
# language standards and link libraries, so their objects get the reference build's flags.
# (No cmake_minimum_required here: it would reset upstream's policy settings for the rest of the scope.)
if(CMAKE_VERSION VERSION_LESS 3.24)
  message(FATAL_ERROR "wiring/hip.cmake needs CMake >= 3.24 (CMAKE_PROJECT_TOP_LEVEL_INCLUDES, DEFER)")
endif()
if(NOT STRATA_ENABLE_HIP OR NOT STRATA_BUILD_DSV41)
  message(FATAL_ERROR "wiring/hip.cmake needs -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_DSV41=ON")
endif()
if(NOT TARGET strata-dsv41 OR NOT TARGET strata_dsv41_kernels)
  message(FATAL_ERROR "upstream did not define strata-dsv41 / strata_dsv41_kernels before the deferred include")
endif()
set(W "${STRATA_WIRING_DIR}")
option(STRATA_FORK_WARM "Build opt-in cross-vendor warm serving" OFF)
if(STRATA_FORK_WARM)
  add_compile_definitions(STRATA_FORK_WARM=1)
endif()
get_filename_component(STAGE "${W}/.." ABSOLUTE)
get_filename_component(SYCLROOT "${W}/../.." ABSOLUTE)
set(STRATA_WIRING_CPU_LANE "stub" CACHE STRING "stub, shared, or a ;-list of sources implementing cpu_lane.hpp")
set(cpu_lane_sources "${W}/cpu_lane_stub.cpp")
if(STRATA_WIRING_CPU_LANE STREQUAL "shared")
  set(cpu_lane_sources "")
elseif(NOT STRATA_WIRING_CPU_LANE STREQUAL "stub")
  set(cpu_lane_sources ${STRATA_WIRING_CPU_LANE})
endif()

get_target_property(up_src strata-dsv41 SOURCES)
set(model_src "")
set(kernel_src "")
foreach(s IN LISTS up_src)
  if(NOT IS_ABSOLUTE "${s}")
    set(s "${CMAKE_SOURCE_DIR}/${s}")
  endif()
  get_filename_component(n "${s}" NAME)
  if(n STREQUAL "ds4_main.cpp")
    continue()
  endif()
  list(APPEND model_src "${s}")
  # the stage DSO needs the device kernels and CPU-lane row code, not Impl / vision / packinfo
  if(NOT n MATCHES "^(ds4_model\\.cu|ds4_vision\\.cu|ds4_packinfo\\.cpp)$")
    list(APPEND kernel_src "${s}")
  endif()
endforeach()

# Copy strata-dsv41's build-relevant properties onto target t (same directory scope, so source-file
# LANGUAGE HIP from upstream's glob applies to the .cu files as well).
macro(strata_wiring_clone t)
  foreach(p INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES LINK_OPTIONS
            CXX_STANDARD CXX_STANDARD_REQUIRED HIP_STANDARD HIP_STANDARD_REQUIRED HIP_ARCHITECTURES
            POSITION_INDEPENDENT_CODE)
    get_target_property(v strata-dsv41 ${p})
    if(NOT v STREQUAL "v-NOTFOUND")
      set_property(TARGET ${t} PROPERTY ${p} "${v}")
    endif()
  endforeach()
endmacro()

# Upstream objects, built with upstream's exact flags (+ PIC for the shared library).
add_library(strata_wiring_upstream_objs OBJECT ${kernel_src})
strata_wiring_clone(strata_wiring_upstream_objs)
set_property(TARGET strata_wiring_upstream_objs PROPERTY POSITION_INDEPENDENT_CODE ON)

set_source_files_properties("${W}/hip_backend.cpp" PROPERTIES LANGUAGE HIP)
set_source_files_properties("${W}/hip_q8k.hip" PROPERTIES LANGUAGE HIP
  COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
add_library(strata-stage-hip-model SHARED
  "${W}/hip_q8k.hip"
  "${STAGE}/abi_plugin.cpp" "${STAGE}/hip_transport.hip"
  "${W}/abi_backend_plugin.cpp" "${W}/hip_backend.cpp" "${W}/pack_source.cpp" ${cpu_lane_sources}
  $<TARGET_OBJECTS:strata_wiring_upstream_objs>)
strata_wiring_clone(strata-stage-hip-model)
# Our own files need C++20 (span, bit_cast); the upstream objects above keep upstream's standard.
set_target_properties(strata-stage-hip-model PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
  HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH_USE_ORIGIN YES BUILD_WITH_INSTALL_RPATH YES INSTALL_RPATH "$ORIGIN")
target_include_directories(strata-stage-hip-model PRIVATE "${W}" "${STAGE}" "${SYCLROOT}" "${CMAKE_SOURCE_DIR}/src/dsv41")
if(STRATA_WIRING_CPU_LANE STREQUAL "shared")
  include("${SYCLROOT}/moe/cpu_shared.cmake")
  target_link_libraries(strata-stage-hip-model PRIVATE strata-row2-cpu)
endif()
target_compile_definitions(strata-stage-hip-model PRIVATE STRATA_STAGE_PLUGIN_HIP=1)
target_compile_options(strata-stage-hip-model PRIVATE -fvisibility=hidden)
target_link_options(strata-stage-hip-model PRIVATE "LINKER:--version-script=${W}/exports_v2.map")

# AMD-only T=1 reference: ref_t1.cpp + every strata-dsv41 source except ds4_main.cpp, upstream flags.
set_source_files_properties("${W}/parity/ref_t1.cpp" PROPERTIES LANGUAGE HIP)
add_executable(strata-dsv41-ref-t1 "${W}/parity/ref_t1.cpp" ${model_src})
strata_wiring_clone(strata-dsv41-ref-t1)
target_include_directories(strata-dsv41-ref-t1 PRIVATE "${CMAKE_SOURCE_DIR}/src/dsv41")
message(STATUS "Strata wiring: HIP stage targets added (strata-stage-hip-model, strata-dsv41-ref-t1)")

if(STRATA_FORK_WARM)
  set_source_files_properties("${W}/../../../warm/hip_dense_check.cpp" PROPERTIES LANGUAGE HIP)
  add_executable(strata-hip-dense-check "${W}/../../../warm/hip_dense_check.cpp"
    $<TARGET_OBJECTS:strata_wiring_upstream_objs>)
  strata_wiring_clone(strata-hip-dense-check)
  target_include_directories(strata-hip-dense-check PRIVATE "${W}/../../../warm" "${CMAKE_SOURCE_DIR}/src/dsv41")
  set_target_properties(strata-hip-dense-check PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
endif()

if(STRATA_WIRING_CPU_LANE STREQUAL "shared")
  set_source_files_properties("${W}/parity/hip_resident_check.cpp" PROPERTIES LANGUAGE HIP)
  add_executable(strata-hip-resident-check "${W}/parity/hip_resident_check.cpp" "${W}/hip_q8k.hip" "${W}/pack_source.cpp"
    $<TARGET_OBJECTS:strata_wiring_upstream_objs>)
  strata_wiring_clone(strata-hip-resident-check)
  target_include_directories(strata-hip-resident-check PRIVATE "${W}" "${STAGE}" "${SYCLROOT}" "${CMAKE_SOURCE_DIR}/src/dsv41")
  target_link_libraries(strata-hip-resident-check PRIVATE strata-row2-cpu)
  target_compile_options(strata-hip-resident-check PRIVATE -fno-fast-math -ffp-contract=off)
  set_target_properties(strata-hip-resident-check PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES BUILD_RPATH_USE_ORIGIN YES
    BUILD_WITH_INSTALL_RPATH YES INSTALL_RPATH "$ORIGIN")
endif()

set_source_files_properties("${W}/parity/fp8_control_hip.cpp" PROPERTIES LANGUAGE HIP)
add_executable(strata-fp8-control-hip "${W}/parity/fp8_control_hip.cpp" $<TARGET_OBJECTS:strata_wiring_upstream_objs>)
strata_wiring_clone(strata-fp8-control-hip)
target_include_directories(strata-fp8-control-hip PRIVATE "${CMAKE_SOURCE_DIR}/src/dsv41")
set_target_properties(strata-fp8-control-hip PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
  HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)

# --- review 2026-11-02 additions (kept apart from row 2's block above so the branches merge cleanly)
if(STRATA_WIRING_CPU_LANE STREQUAL "shared")
  # Q5_K embedding decode vs upstream gg_row_to_bf16 on every row (CPU only; freezes q5k_fma).
  add_executable(strata-wiring-embed-check "${W}/parity/embed_check.cpp" "${W}/pack_source.cpp")
  target_include_directories(strata-wiring-embed-check PRIVATE "${W}" "${STAGE}" "${CMAKE_SOURCE_DIR}/src/dsv41" ${row2_includes})
  set_target_properties(strata-wiring-embed-check PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
  target_link_libraries(strata-wiring-embed-check PRIVATE strata-row2-cpu)
endif()
# Our Q5_K decoder selects contraction explicitly (q5k_fma); the compiler must not contract it on its own.
set_source_files_properties("${W}/pack_source.cpp" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
