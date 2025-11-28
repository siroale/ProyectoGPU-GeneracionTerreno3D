# ProyectoGPU-GeneracionTerreno3D
Generación de terreno 3D usando Perlin Noise

# Instalar CUDA y librerias de Opengl
CUDA:
```
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install build-essential cmake -y
sudo apt-get install nvidia-cuda-toolkit -y
```

Librerias de OpenGL:
```
sudo apt-get install libglfw3-dev libglew-dev libglm-dev -y
```
# Compilación
Asegurarse de estar dentro de la carpeta `/build`, y ejecutar el comando `make`

Ejecutar con `./TerrenoCUDA`

# Configuración importante:
En el archivo de configuración `CMakeLists.txt` en la linea 9, se debe ajustar el número segun la tarjeta gráfica que tengas.
RTX 4000 series (4060, 4070...): Cambiar por 89

RTX 3000 series (3060, 3070...): Cambiar por 86

RTX 2000 series (2060, 2070...): Cambiar por 75

GTX 1600 series (1660, 1650...): Cambiar por 75

GTX 1000 series (1060, 1070...): Cambiar por 61
