#pragma once
#include <cuda_runtime.h>

// Resolución del grid volumétrico
#define GRID_SIZE 128
#define TOTAL_MAX_VERTS (GRID_SIZE * GRID_SIZE * GRID_SIZE * 5 * 3)

struct TerrainParams {
    float time;
    float globalX, globalZ;
    
    // Parámetros recuperados del original
    float waterLevel;    // Nivel del mar
    float heightMult;    // Altura de montañas
    float scale;         // Zoom general
    int   octaves;       // Detalle del ruido
    int   enableShadows; // Checkbox de sombras
    
    // Iluminación
    float3 sunDir;
    float3 skyColor;
    float isolevel;      // Para controlar el "grosor" del terreno base
};

struct Vertex {
    float x, y, z, w;
    float r, g, b, a;
};

void initCudaMemory();
void cleanupCudaMemory();
int runCudaKernel(Vertex* host_ptr, TerrainParams params);