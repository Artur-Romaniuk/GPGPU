cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

project(Assignment)

function(add_shader TARGET SHADER)
    find_program(GLSLC glslc)

    set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER})
    set(current-output-path ${CMAKE_CURRENT_SOURCE_DIR}/build/${SHADER}.spv)

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})

    add_custom_command(
           OUTPUT ${current-output-path}
           COMMAND ${GLSLC} --target-env=vulkan1.2 -o ${current-output-path} ${current-shader-path}
           DEPENDS ${current-shader-path}
           IMPLICIT_DEPENDS CXX ${current-shader-path}
           VERBATIM)

    # Make sure our build depends on this output.
    set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${current-output-path})
endfunction(add_shader)

add_subdirectory(libs)
find_package(Vulkan REQUIRED)

set(RENDERDOC_PATH /usr/include)

### Don't forget that there is a CMakeLists.txt in libs ! ###

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER SOURCE_FILES EXCLUDE REGEX "_teach.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX "_teach.h$")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ./include)
target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS} )#libs/glfw/include)
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan stb csv)#vma glfw)
target_compile_definitions(${PROJECT_NAME} PRIVATE)# GLFW_INCLUDE_NONE)
if(RENDERDOC_PATH)
    target_include_directories(${PROJECT_NAME} PRIVATE ${RENDERDOC_PATH})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_RENDERDOC)
endif()

file(GLOB_RECURSE SHADER_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")
list(FILTER SHADER_FILES EXCLUDE REGEX "_teach.comp$")
foreach(SHADER_FILE ${SHADER_FILES})
    add_shader(${PROJECT_NAME} ${SHADER_FILE})
endforeach()
