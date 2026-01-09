cmake_minimum_required(VERSION 3.12)
project(example)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if (MSVC)
    add_compile_options(/Zc:__cplusplus)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")    
endif()
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use Imgui"          ON)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)

set(LIBIGL_DIR "external/libigl/build")
include(cmake/FindLIBIGL.cmake)
set(PortAudio_DIR "C:/Program Files (x86)/PortAudio")
find_package(PortAudio CONFIG REQUIRED)
add_subdirectory(external/protobuf protobuf)

# Add your project files
file(GLOB SRCFILES *.cpp *.cc)
file(GLOB TOOLS tools/*.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_INCLUDE_DIRS} external/portaudio/include external/protobuf/src)
add_subdirectory(external/HouGeoIO)
foreach(TOOL_NAME ${TOOLS})
    get_filename_component(TOOL_BIN ${TOOL_NAME} NAME_WE)
    set(TOOL_BIN ${TOOL_BIN}_bin)
    message("-- Building tool : ${TOOL_BIN}")
    add_executable(${TOOL_BIN} ${SRCFILES} ${TOOL_NAME})
    target_link_libraries(${TOOL_BIN}
        igl::core
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        igl::png
        HouGeoIO
        ${PORTAUDIO_LIBRARIES}
        protobuf::libprotobuf)
    target_include_directories(${TOOL_BIN} PRIVATE external/HouGeoIO/src external/HouGeoIO/extern)
endforeach()
