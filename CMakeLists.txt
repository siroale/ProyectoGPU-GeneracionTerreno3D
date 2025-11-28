cmake_minimum_required(VERSION 3.18)
project(TerrenoCUDA LANGUAGES CXX CUDA) # Activamos C++ (CXX) y CUDA

# Configuración de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuración de CUDA
set(CMAKE_CUDA_ARCHITECTURES 86) # -----> EL NUMERO SE CAMBIA DEPENDIENDO DE QUE GRAFICA TIENES!!

# Buscar paquetes de OpenGL y Ventanas
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Buscar las librerías de CUDA (necesarias para Interop)
find_package(CUDAToolkit REQUIRED)

# Definir los archivos fuente
set(SOURCES 
    main.cpp
    kernel.cu
)

# Crear el ejecutable
add_executable(${PROJECT_NAME} ${SOURCES})

# Unimos nuestro ejecutable con las librerías necesarias
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart  # Runtime de CUDA
)

# Incluir directorios (para que encuentre kernel.h y headers de CUDA)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)