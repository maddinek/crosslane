# Injected into the UPSTREAM top-level configure with -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=<this file>.
# CMake includes it right after upstream's first project(); the deferred include runs at the end of the
# top-level directory, when strata-dsv41 / strata_dsv41_kernels and their usage requirements exist.
get_filename_component(STRATA_WIRING_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
cmake_language(DEFER CALL include "${STRATA_WIRING_DIR}/hip.cmake")
