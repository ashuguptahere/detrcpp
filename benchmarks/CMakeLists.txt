# Microbenchmarks (Google Benchmark). Real benchmark targets land in Phase 3
# (`--benchmark` / model table). This stub keeps DETR_BUILD_BENCHMARKS=ON
# configurable today without a dangling add_subdirectory.

find_package(benchmark CONFIG REQUIRED)

# Example wiring for when benchmarks exist:
#   add_executable(detr_bench_boxops boxops_bench.cpp)
#   target_link_libraries(detr_bench_boxops PRIVATE detr::core benchmark::benchmark detr_warnings)
#   set_target_properties(detr_bench_boxops PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)

message(STATUS "detrcpp: benchmarks enabled (no benchmark targets defined yet — Phase 3).")
