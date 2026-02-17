cmake_minimum_required(VERSION 3.16)
project(executorch_vision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTORCH_VISION_CLANG_TIDY_RUN_ARGS "-p=${CMAKE_BINARY_DIR}")

include(FetchContent)

option(EXECUTORCH_VISION_ENABLE_TESTS "Enable building tests (requires GoogleTest)." ON)

add_library(executorch_vision_lib src/executorch_vision.cpp)
target_include_directories(executorch_vision_lib PUBLIC include)
target_compile_features(executorch_vision_lib PUBLIC cxx_std_17)

add_executable(executorch_vision_bin src/main.cpp)
target_link_libraries(executorch_vision_bin PRIVATE executorch_vision_lib)

if(EXECUTORCH_VISION_ENABLE_TESTS)
  enable_testing()

  # Prefer system-installed GTest (via package manager). If not found, fall back to FetchContent.
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    # Use timestamping for fetch/ extract and specify URL hash for robustness.
    set(FETCHCONTENT_QUIET OFF)
    set(FETCHCONTENT_UPDATES_DISCONNECTED OFF)
    message(STATUS "GTest not found: Fetching googletest via FetchContent (git)")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG main
    )
  FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(executorch_vision_test tests/executorch_vision_test.cpp)
  # Link against the googletest target provided by either the package config or the
  # FetchContent subproject. Prefer the imported target if present.
  if(TARGET GTest::gtest_main)
    target_link_libraries(executorch_vision_test PRIVATE executorch_vision_lib GTest::gtest_main)
  else()
    target_link_libraries(executorch_vision_test PRIVATE executorch_vision_lib gtest_main)
  endif()
  add_test(NAME executorch-vision-test COMMAND executorch_vision_test)
else()
  message(STATUS "EXECUTORCH_VISION_ENABLE_TESTS is OFF: tests will not be built")
endif()

# Formatting target using clang-format
find_program(CLANG_FORMAT NAMES clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_SOURCE_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/include/*.h
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/tests/*.cpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCE_FILES}
    COMMENT "Formatting source files with clang-format"
  )
endif()

# Clang-tidy configuration
find_program(CLANG_TIDY NAMES clang-tidy)
if(CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY};-checks=*, -llvm-header-guard")

  # Add a convenience target to run clang-tidy over all source files.
  file(GLOB_RECURSE EXECUTORCH_VISION_SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/include/*.h)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY} ${EXECUTORCH_VISION_SRC_FILES} -- ${EXECUTORCH_VISION_CLANG_TIDY_RUN_ARGS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-tidy over project sources"
  )
endif()
