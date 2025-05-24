# find_package(OpenCV 4 REQUIRED)

# find_library(ACCELERATE Accelerate REQUIRED)
# find_library(METAL Metal REQUIRED)
# find_library(FOUNDATION Foundation REQUIRED)



# add_library(bridge_cv OBJECT ${BRIDGE_DIR}/include/bridge.h ${BRIDGE_DIR}/lib/bridge.cpp)

# target_link_directories(bridge_cv PRIVATE ${LIBTORCH_DIR}/lib)

# target_link_libraries(
#     bridge_cv
#     PRIVATE
#     -ltorch
#     -ltorch_cpu
#     -lc10
#     -ltorch_global_deps
#     ${OpenCV_LIBS}
#     # ${TORCH_LIBRARIES}
#     ${ACCELERATE}
#     ${METAL}
#     ${FOUNDATION}
# )

# target_include_directories(
#     bridge_cv
#     PRIVATE
#     ${BRIDGE_DIR}/include
#     ${LIBTORCH_DIR}/include
#     ${LIBTORCH_DIR}/include/torch/csrc/api/include
#     # ${BRIDGE_DIR}/util
# )

# # if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# #   target_compile_options(bridge_cv PRIVATE -Ofast -flto -ffast-math)
# #   target_link_options(bridge_cv PRIVATE -flto)
# # endif()


# set(BRIDGE_CV_OBJECT_FILES $<TARGET_OBJECTS:bridge_cv>)


# set(CHAI_CV_LINKER_ARGS
#     -M ${PROJECT_ROOT_DIR}/lib
#     ${BRIDGE_DIR}/include/bridge.h
#     ${BRIDGE_CV_OBJECT_FILES}
#     -L ${LIBTORCH_DIR}/lib
#     ${LIBTORCH_LIBS_LINKER_ARGS}
#     --ldflags "-Wl,-rpath,${LIBTORCH_DIR}/lib"
# )








# add_executable(SplitLoop 
#     ${CMAKE_CURRENT_SOURCE_DIR}/split_loop.chpl
#     ${CHAI_LIB_FILES}
# )

# add_dependencies(SplitLoop bridge_cv)
# # add_dependencies(SplitLoop ChAI)
# target_link_options(SplitLoop
#     PRIVATE
#     ${CHAI_CV_LINKER_ARGS}
# )

# cmake_print_variables(CHAI_CV_LINKER_ARGS)
# cmake_print_variables(OpenCV_LIBS)
# cmake_print_variables(ACCELERATE)
# cmake_print_variables(METAL)
# cmake_print_variables(FOUNDATION)

# set_target_properties(SplitLoop PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# # if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# #   target_compile_options(SplitLoop PRIVATE -Ofast -flto -ffast-math)
# #   target_link_options(SplitLoop PRIVATE -flto)
# # endif()