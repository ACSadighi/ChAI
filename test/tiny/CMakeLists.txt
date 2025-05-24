add_executable(AvenNllTest 
    ${CMAKE_CURRENT_SOURCE_DIR}/aven_nll_test.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(AvenNllTest bridge)
add_dependencies(AvenNllTest ChAI)
target_link_options(AvenNllTest
    PRIVATE
    ${CHAI_LINKER_ARGS}
)

set_target_properties(AvenNllTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

