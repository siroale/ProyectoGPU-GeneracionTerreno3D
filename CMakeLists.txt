cmake_minimum_required(VERSION 3.18)
project(TerrenoCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75) 

# --- DEPENDENCIAS ---
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --- DESCARGAR IMGUI AUTOMÁTICAMENTE ---
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG        docking # Usamos la rama 'docking' que es la más moderna
)
FetchContent_MakeAvailable(imgui)

# Definir fuentes de ImGui necesarias para OpenGL y GLFW
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# --- TU PROYECTO ---
set(SOURCES 
    main.cpp
    cpu_kernel.cpp
    ${IMGUI_SOURCES} # Agregamos ImGui a tu ejecutable
)

add_executable(${PROJECT_NAME} ${SOURCES})

# Incluir directorios
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${imgui_SOURCE_DIR}             # Para encontrar imgui.h
    ${imgui_SOURCE_DIR}/backends    # Para encontrar los backends
)

# Linkear librerías
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
)