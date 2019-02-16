set(TESTNAME test_poseGenerator)
sdk_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(
    ${SDK_BINARY_DIR}/configured/tests)

set(LIBRARIES
    gtest
    projPoseGenerator
)

set(SOURCES
    main.cpp
    TestPoseGenerator.cpp
)

sdk_add_test(${TESTNAME} "${SOURCES}" "${LIBRARIES}")

