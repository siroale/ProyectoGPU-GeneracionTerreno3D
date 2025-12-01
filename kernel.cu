#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h> 
#include <stdio.h>
#include "kernel.h"

Vertex* d_vertices = nullptr;
TreeInstance* d_trees = nullptr;
int* d_tree_count = nullptr;

// --- MATEMÁTICAS ---
__device__ float fract(float x) { return x - floorf(x); }
__device__ float hash(float x, float y) { return fract(sinf(x * 12.9898f + y * 78.233f) * 43758.5453123f); }
__device__ float lerp(float a, float b, float t) { return a + t * (b - a); }
__device__ float clamp(float v, float minVal, float maxVal) { return fmaxf(minVal, fminf(v, maxVal)); }

__device__ float noise(float x, float y) {
    float ix = floorf(x); float iy = floorf(y);
    float fx = x - ix; float fy = y - iy;
    float a = hash(ix, iy); float b = hash(ix + 1.0f, iy);
    float c = hash(ix, iy + 1.0f); float d = hash(ix + 1.0f, iy + 1.0f);
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    return lerp(lerp(a, b, ux), lerp(c, d, ux), uy);
}

__device__ float fbm(float x, float y, int octaves) {
    float total = 0.0f; float amp = 0.5f; float freq = 1.0f; float maxV = 0.0f;
    for(int i=0; i<octaves; i++) { total += noise(x*freq, y*freq)*amp; maxV+=amp; amp*=0.5f; freq*=2.0f; }
    return total/maxV;
}

__device__ float ridge_noise(float x, float y, int octaves) {
    float total = 0.0f; float amp = 1.0f; float freq = 1.0f; float maxV = 0.0f;
    for(int i=0; i<octaves; i++) { 
        float n = 1.0f - fabs(noise(x*freq, y*freq)); 
        n*=n; total+=n*amp; maxV+=amp; amp*=0.5f; freq*=2.0f; 
    }
    return total/maxV;
}

__device__ void normalize(float3 &v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if(len > 0.0f){ v.x/=len; v.y/=len; v.z/=len; }
}
__device__ float dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

// --- CÁLCULO DE SOMBRAS PROYECTADAS ---
__device__ float calculateShadow(float3 pos, float3 sunDir, TerrainParams p) {
    if (p.enableShadows == 0) return 1.0f;

    float shadow = 1.0f;
    float stepSize = 0.08f * p.scale;
    float3 currentPos = pos;
    
    for (int i = 1; i <= 16; i++) {
        currentPos.x += sunDir.x * stepSize;
        currentPos.y += sunDir.y * stepSize;
        currentPos.z += sunDir.z * stepSize;

        if (currentPos.y > p.heightMult) break;

        float h = ridge_noise(currentPos.x, currentPos.z, 2);
        float groundY = h * p.heightMult - 0.5f;

        if (currentPos.y < groundY) {
            shadow = 0.2f;
            break;
        }
    }
    return shadow;
}

// --- KERNEL DEL TERRENO (SIN ÁRBOLES) ---
__global__ void terrain_kernel(Vertex* vertices, unsigned int width, unsigned int height, TerrainParams p) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned int idx = y * width + x;

        float u = x / (float)width;
        float v = y / (float)height;
        float cx = u * 2.0f - 1.0f;
        float cy = v * 2.0f - 1.0f;

        float worldX = (cx + p.globalX) * p.scale;
        float worldZ = (cy + p.globalZ) * p.scale;

        float hBase = ridge_noise(worldX, worldZ, p.octaves);
        float finalY = hBase * p.heightMult - 0.5f;
        float humidity = fbm(worldX*0.5f + 50.0f, worldZ*0.5f + 50.0f, 4);

        float eps = 0.01f; 
        float h_r = ridge_noise(worldX + eps * p.scale, worldZ, p.octaves) * p.heightMult - 0.5f;
        float h_d = ridge_noise(worldX, worldZ + eps * p.scale, p.octaves) * p.heightMult - 0.5f;
        float3 normal = {finalY - h_r, eps, finalY - h_d};
        normalize(normal);

        bool isWater = finalY < p.waterLevel;
        if (isWater) {
            finalY = p.waterLevel;
            normal = {0.0f, 1.0f, 0.0f};
        }

        float diff = fmaxf(dot(normal, p.sunDir), 0.0f);
        
        float shadowVal = 1.0f;
        if (diff > 0.0f && !isWater) {
            shadowVal = calculateShadow({worldX, finalY, worldZ}, p.sunDir, p);
        }

        float ambient = 0.15f; 
        float lightIntensity = clamp(ambient + diff * shadowVal, 0.0f, 1.0f);

        float r, g, b;

        if (isWater) {
            r = p.skyColor.x * 0.5f; 
            g = p.skyColor.y * 0.5f + 0.1f; 
            b = p.skyColor.z * 0.8f + 0.2f;
            lightIntensity = 0.8f + diff * 0.4f;
        } else {
            float hRel = clamp((finalY - p.waterLevel) / (p.heightMult - p.waterLevel), 0.0f, 1.0f);
            if (hRel < 0.05f) { r=0.8f; g=0.75f; b=0.5f; }
            else if (hRel > 0.75f) { r=0.95f; g=0.95f; b=1.0f; }
            else {
                if (humidity < 0.45f) { r=0.6f; g=0.5f; b=0.3f; }
                else { r=0.1f; g=0.5f; b=0.1f; }
            }
        }

        if (p.sunDir.y < 0.2f && p.sunDir.y > -0.2f) {
            float sunsetFactor = 1.0f - abs(p.sunDir.y) * 5.0f;
            r += sunsetFactor * 0.3f;
            g -= sunsetFactor * 0.1f;
        }

        r *= lightIntensity; g *= lightIntensity; b *= lightIntensity;

        float dist = sqrtf(cx*cx + cy*cy);
        float fogFactor = clamp((dist - 0.75f) / 0.5f, 0.0f, 1.0f);
        r = lerp(r, p.skyColor.x, fogFactor);
        g = lerp(g, p.skyColor.y, fogFactor);
        b = lerp(b, p.skyColor.z, fogFactor);

        vertices[idx] = { cx, finalY, cy, 1.0f, r, g, b, 1.0f };
    }
}

// --- NUEVO: KERNEL PARA GENERAR INSTANCIAS DE ÁRBOLES ---
__global__ void tree_kernel(TreeInstance* trees, int* count, unsigned int width, unsigned int height, TerrainParams p) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = x / (float)width;
        float v = y / (float)height;
        float cx = u * 2.0f - 1.0f;
        float cy = v * 2.0f - 1.0f;

        float worldX = (cx + p.globalX) * p.scale;
        float worldZ = (cy + p.globalZ) * p.scale;

        float hBase = ridge_noise(worldX, worldZ, p.octaves);
        float finalY = hBase * p.heightMult - 0.5f;
        float humidity = fbm(worldX*0.5f + 50.0f, worldZ*0.5f + 50.0f, 4);

        bool isWater = finalY < p.waterLevel;
        
        float hRel = clamp((finalY - p.waterLevel) / (p.heightMult - p.waterLevel), 0.0f, 1.0f);
        bool canHaveTree = !isWater && hRel >= 0.05f && hRel <= 0.75f && humidity >= 0.45f;

        float treeProb = hash(worldX * 60.0f, worldZ * 60.0f);
        
        if (canHaveTree && treeProb > 0.985f) {
            int idx = atomicAdd(count, 1);
            if (idx < MAX_TREES) {
                // Calcular iluminación para el árbol (más pronunciada que el terreno)
                float3 normal = {0.0f, 1.0f, 0.0f};
                float diff = fmaxf(dot(normal, p.sunDir), 0.0f);
                float shadowVal = calculateShadow({worldX, finalY, worldZ}, p.sunDir, p);
                
                // Iluminación más contrastada para árboles (más ambient light)
                float ambient = 0.25f; // Más luz ambiente que el terreno
                float lightIntensity = clamp(ambient + diff * shadowVal * 1.2f, 0.0f, 1.0f);
                
                // Color verde del árbol con iluminación más visible
                float tr = 0.15f * lightIntensity;
                float tg = 0.6f * lightIntensity;  // Verde más saturado
                float tb = 0.15f * lightIntensity;
                
                // Aplicar niebla
                float dist = sqrtf(cx*cx + cy*cy);
                float fogFactor = clamp((dist - 0.75f) / 0.5f, 0.0f, 1.0f);
                tr = lerp(tr, p.skyColor.x, fogFactor);
                tg = lerp(tg, p.skyColor.y, fogFactor);
                tb = lerp(tb, p.skyColor.z, fogFactor);
                
                // Escala aleatoria para variedad
                float treeScale = 0.8f + hash(worldX * 123.4f, worldZ * 567.8f) * 0.4f;
                
                trees[idx] = {cx, finalY, cy, treeScale, tr, tg, tb};
            }
        }
    }
}

// Host functions
void initCudaMemory() { 
    cudaMalloc((void**)&d_vertices, NUM_TERRAIN_POINTS * sizeof(Vertex));
    cudaMalloc((void**)&d_trees, MAX_TREES * sizeof(TreeInstance));
    cudaMalloc((void**)&d_tree_count, sizeof(int));
}

void cleanupCudaMemory() { 
    if (d_vertices) cudaFree(d_vertices);
    if (d_trees) cudaFree(d_trees);
    if (d_tree_count) cudaFree(d_tree_count);
}

void runCudaKernel(Vertex* host_ptr, TerrainParams params) {
    if (!d_vertices) return;
    dim3 b(16, 16);
    dim3 g((MESH_WIDTH + b.x - 1) / b.x, (MESH_HEIGHT + b.y - 1) / b.y);
    terrain_kernel<<<g, b>>>(d_vertices, MESH_WIDTH, MESH_HEIGHT, params);
    cudaMemcpy(host_ptr, d_vertices, NUM_TERRAIN_POINTS * sizeof(Vertex), cudaMemcpyDeviceToHost);
}

void runTreeKernel(TreeInstance* host_trees, int* tree_count, TerrainParams params) {
    if (!d_trees || !d_tree_count) return;
    
    // Resetear el contador
    int zero = 0;
    cudaMemcpy(d_tree_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Ejecutar kernel de árboles
    dim3 b(16, 16);
    dim3 g((MESH_WIDTH + b.x - 1) / b.x, (MESH_HEIGHT + b.y - 1) / b.y);
    tree_kernel<<<g, b>>>(d_trees, d_tree_count, MESH_WIDTH, MESH_HEIGHT, params);
    
    // Copiar resultados
    cudaMemcpy(tree_count, d_tree_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_trees, d_trees, (*tree_count) * sizeof(TreeInstance), cudaMemcpyDeviceToHost);
}