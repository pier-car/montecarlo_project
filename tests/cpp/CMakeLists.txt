include(FetchContent)

# Prefer a system-installed GoogleTest; fall back to fetching a pinned release.
find_package(GTest QUIET)

if(NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(nephele_tests
  test_wellmixed.cpp
  test_grid.cpp
  test_enkf.cpp
)
target_link_libraries(nephele_tests PRIVATE nephele_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(nephele_tests)
