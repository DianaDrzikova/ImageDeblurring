project(stochastic_deconvolution_sample)

cmake_minimum_required(VERSION 3.10)
project(StochasticDeconvolution)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(stochastic_deconvolution_sample
    main.cpp
    stochastic_deconvolution.cpp
    regularizers.cpp
)


target_link_libraries(stochastic_deconvolution_sample ${OpenCV_LIBS})

set_target_properties(stochastic_deconvolution_sample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_SOURCE_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_SOURCE_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/../bin
)
