cmake_minimum_required(VERSION 3.24)
project(tfmbs_mlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(NOT LLVM_DIR)
  set(LLVM_DIR "/Users/t81dev/Code/llvm-project/build/lib/cmake/llvm")
endif()

if(NOT MLIR_DIR)
  set(MLIR_DIR "/Users/t81dev/Code/llvm-project/build/lib/cmake/mlir")
endif()

list(APPEND CMAKE_PREFIX_PATH ${LLVM_DIR} ${MLIR_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR} ${MLIR_DIR})

find_package(MLIR REQUIRED CONFIG)
include("${LLVM_DIR}/AddLLVM.cmake")
include("${MLIR_DIR}/AddMLIR.cmake")

add_mlir_library(tfmbs_dialect SHARED
  src/mlir/TfmbsDialect.cpp
  src/mlir/TfmbsPasses.cpp
  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRAffineDialect
    MLIRLinalgDialect
)

target_include_directories(tfmbs_dialect PUBLIC ${MLIR_INCLUDE_DIRS})
target_include_directories(tfmbs_dialect PUBLIC ${LLVM_INCLUDE_DIRS})

mlir_check_all_link_libraries(tfmbs_dialect)

add_executable(TfmbsPassTest tests/mlir/TfmbsPassTest.cpp)
target_include_directories(TfmbsPassTest PRIVATE src/mlir)
target_link_libraries(TfmbsPassTest PRIVATE tfmbs_dialect)

add_lit_testsuite(check-tfmbs "Running TFMBS MLIR regression tests"
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/lit.cfg
  PARAMS BINARY_DIR=${CMAKE_BINARY_DIR}
  DEPENDS TfmbsPassTest
)

add_mlir_library(tfmbs_plugin SHARED
  src/mlir/TfmbsPassPlugin.cpp
  LINK_LIBS PUBLIC
    tfmbs_dialect
    MLIRIR
    MLIRPass
    MLIRSupport
)

set_target_properties(tfmbs_dialect PROPERTIES
  INSTALL_RPATH "@loader_path;@loader_path/../../../lib;/Users/t81dev/Code/llvm-project/build-shared/lib")
set_target_properties(tfmbs_plugin PROPERTIES
  INSTALL_RPATH "@loader_path;@loader_path/../../../lib;/Users/t81dev/Code/llvm-project/build-shared/lib")
