#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "kernel.h"

Vertex* d_vertices = nullptr;

// --- FUNCIONES MATEMÁTICAS ---
__device__ float fract(float x) { return x - floorf(x); }
__device__ float hash(float x, float y) { return fract(sinf(x * 12.9898f + y * 78.233f) * 43758.5453123f); }
__device__ float lerp(float a, float b, float t) { return a + t * (b - a); }

__device__ float noise(float x, float y) {
    float ix = floorf(x); float iy = floorf(y);
    float fx = x - ix; float fy = y - iy;
    float a = hash(ix, iy); float b = hash(ix + 1.0f, iy);
    float c = hash(ix, iy + 1.0f); float d = hash(ix + 1.0f, iy + 1.0f);
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    return lerp(lerp(a, b, ux), lerp(c, d, ux), uy);
}

// Terreno default
__device__ float fbm(float x, float y) {
    float total = 0.0f; float amplitude = 1.0f; float frequency = 1.0f; float maxValue = 0.0f;
    for(int i = 0; i < 6; i++) {
        total += noise(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude; amplitude *= 0.5f; frequency *= 2.0f;
    }
    return total / maxValue;
}


// Islas insanas
__device__ float ridge_noise(float x, float y) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;

    for(int i = 0; i < 6; i++) {
        float n = 1.0f - fabs(noise(x * frequency, y * frequency));
        n = n * n; 
        total += n * amplitude;
        maxValue += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }

    return total / maxValue;
}


// --- KERNEL PRINCIPAL ---
__global__ void terrain_kernel(Vertex* vertices, unsigned int width, unsigned int height, float time) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Índices:
        unsigned int terrainIndex = y * width + x;
        // El punto del árbol correspondiente está en la segunda mitad del buffer
        unsigned int treeIndex = terrainIndex + NUM_TERRAIN_POINTS;

        float u = x / (float)width;
        float v = y / (float)height;

        // Generar Altura
        float noiseX = u * 4.0f;
        float noiseZ = v * 4.0f;
        float heightVal = ridge_noise(noiseX, noiseZ + time);
        float finalY = heightVal * 1.2f - 0.6f;

        // --- COLOREADO DEL TERRENO ---
        float r, g, b;
        bool canHaveTree = false;

        if (heightVal < 0.35f) { r = 0.0f; g = 0.1f; b = 0.8f; } // Agua
        else if (heightVal < 0.40f) { r = 0.9f; g = 0.8f; b = 0.6f; } // Playa
        else if (heightVal < 0.65f) { 
            r = 0.1f; g = 0.6f; b = 0.1f; // Pasto
            canHaveTree = true;
        } 
        else if (heightVal < 0.85f) { r = 0.5f; g = 0.5f; b = 0.5f; } // Roca
        else { r = 1.0f; g = 1.0f; b = 1.0f; } // Nieve

        // 1. ESCRIBIR PUNTO DE TERRENO
        vertices[terrainIndex] = {
            u * 2.0f - 1.0f, finalY, v * 2.0f - 1.0f, 1.0f, // Posición
            r, g, b, 1.0f                                   // Color
        };

        // 2. ESCRIBIR PUNTO DE ÁRBOL
        float treeProb = hash(u * 100.0f, v * 100.0f); // Probabilidad aleatoria

        // Si es zona de pasto hay probabilidad de que ponga un árbol
        if (canHaveTree && treeProb > 0.99f) {
            vertices[treeIndex] = {
                u * 2.0f - 1.0f, finalY + 0.05f, v * 2.0f - 1.0f, 1.0f, // Posición: Un poco más arriba (+0.05f)
                0.0f, 0.3f, 0.0f, 1.0f                                  // Color: Verde oscuro
            };
        } else {
            // Si no hay árbol, hacemos este punto invisible (Alpha = 0)
            vertices[treeIndex] = { 0,0,0,0,  0,0,0,0.0f };
        }
    }
}

// --- GESTIÓN DE MEMORIA ---
void initCudaMemory() {
    // IMPORTANTE: Reservamos el DOBLE de memoria
    size_t size = TOTAL_VERTICES * sizeof(Vertex);
    cudaMalloc((void**)&d_vertices, size);
}

void cleanupCudaMemory() {
    if (d_vertices) cudaFree(d_vertices);
}

void runCudaKernel(Vertex* host_ptr, float time) {
    if (!d_vertices) return;
    dim3 blockSize(16, 16);
    // Seguimos iterando sobre la cuadrícula base (MESH_WIDTH x MESH_HEIGHT)
    dim3 gridSize((MESH_WIDTH + blockSize.x - 1) / blockSize.x, 
                  (MESH_HEIGHT + blockSize.y - 1) / blockSize.y);

    terrain_kernel<<<gridSize, blockSize>>>(d_vertices, MESH_WIDTH, MESH_HEIGHT, time);
    
    // Copiamos el buffer TOTAL
    size_t size = TOTAL_VERTICES * sizeof(Vertex);
    cudaMemcpy(host_ptr, d_vertices, size, cudaMemcpyDeviceToHost);
}