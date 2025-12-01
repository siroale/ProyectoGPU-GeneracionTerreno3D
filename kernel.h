#pragma once

#include <cuda_runtime.h>

#define MESH_WIDTH  512
#define MESH_HEIGHT 512
#define NUM_TERRAIN_POINTS (MESH_WIDTH * MESH_HEIGHT)
#define TOTAL_VERTICES (NUM_TERRAIN_POINTS * 2)

// Número de triángulos: cada quad = 2 triángulos, y hay (W-1)*(H-1) quads
#define NUM_QUADS ((MESH_WIDTH - 1) * (MESH_HEIGHT - 1))
#define NUM_INDICES (NUM_QUADS * 6)  // 6 índices por quad (2 triángulos)

struct TerrainParams {
    float time;
    float globalX;
    float globalZ;
    float waterLevel;
    float scale;
    float heightMult;
    int   octaves;
    
    // Parámetros de iluminación
    float3 sunDir;    
    float3 skyColor;  
    int   enableShadows; 
};

struct Vertex {
    float x, y, z, w;
    float r, g, b, a;
};

void initCudaMemory();
void cleanupCudaMemory();
void runCudaKernel(Vertex* host_ptr, TerrainParams params);