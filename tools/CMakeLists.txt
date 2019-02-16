#-------------------------------------------------------------------------------
# cmake project
#-------------------------------------------------------------------------------
project(projPoseGenerator)

#-------------------------------------------------------------------------------
# Enable auto formatting and compiler flags
#-------------------------------------------------------------------------------
sdk_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
include(SDKConfiguration)

add_library(${PROJECT_NAME}
    src/poseGenerator.cpp
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        projDataSampler
        projAugmenter
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
      "${PROJECT_SOURCE_DIR}/include"
)

