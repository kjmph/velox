# End-to-end test for the UCX exchange with transport discovery.
#
# Deliberately not a test target: no add_test(), so it never runs in ctest. It
# needs a UCX listener and a free port, which is not something a unit test
# suite should be arranging.
#
# Requires VELOX_BUILD_TESTING for PlanBuilder, and VELOX_BUILD_LAB to be
# switched on, which it is not by default. cuDF is required rather than
# optional: every scenario is about where the data lives, so the program packs
# tables and asks CUDA which side of the bus a pointer is on.
if(VELOX_ENABLE_UCX_EXCHANGE AND VELOX_ENABLE_CUDF)
  add_executable(velox_lab_dynamic_ucx DynamicUcxLab.cpp)

  target_link_libraries(
    velox_lab_dynamic_ucx
    PRIVATE
      velox_ucx_exchange
      velox_exec
      velox_exec_test_lib
      velox_presto_serializer
      velox_functions_prestosql
      velox_vector_test_lib
      velox_cudf_exec
      cudf::cudf
      CUDA::cudart
      Folly::folly
      fmt::fmt
      ucxx::ucxx
  )
endif()
