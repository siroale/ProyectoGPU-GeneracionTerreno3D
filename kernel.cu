#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h> 
#include <stdio.h>
#include "kernel.h"

Vertex* d_vertices = nullptr;

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

// --- NUEVO: CÁLCULO DE SOMBRAS PROYECTADAS (Raymarching) ---
__device__ float calculateShadow(float3 pos, float3 sunDir, TerrainParams p) {
    if (p.enableShadows == 0) return 1.0f;

    float shadow = 1.0f;
    float stepSize = 0.08f * p.scale; // Paso del rayo
    float3 currentPos = pos;
    
    // Lanzamos un rayo hacia el sol y miramos si chocamos con algo
    for (int i = 1; i <= 16; i++) {
        currentPos.x += sunDir.x * stepSize;
        currentPos.y += sunDir.y * stepSize;
        currentPos.z += sunDir.z * stepSize;

        // Si el rayo se va muy alto, ya no choca con nada
        if (currentPos.y > p.heightMult) break;

        // Calculamos la altura del terreno en ese punto del rayo
        float h = ridge_noise(currentPos.x, currentPos.z, 2); // Usamos menos octaves (2) para que sea rápido
        float groundY = h * p.heightMult - 0.5f;

        // Si el rayo está por debajo de la tierra -> Sombra
        if (currentPos.y < groundY) {
            shadow = 0.2f; // Zona oscura (no negro total para simular luz ambiente)
            break;
        }
    }
    return shadow;
}

__global__ void terrain_kernel(Vertex* vertices, unsigned int width, unsigned int height, TerrainParams p) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned int idx = y * width + x;
        unsigned int treeIdx = idx + NUM_TERRAIN_POINTS;

        float u = x / (float)width;
        float v = y / (float)height;
        float cx = u * 2.0f - 1.0f;
        float cy = v * 2.0f - 1.0f;

        float worldX = (cx + p.globalX) * p.scale;
        float worldZ = (cy + p.globalZ) * p.scale;

        // Altura y Humedad
        float hBase = ridge_noise(worldX, worldZ, p.octaves);
        float finalY = hBase * p.heightMult - 0.5f;
        float humidity = fbm(worldX*0.5f + 50.0f, worldZ*0.5f + 50.0f, 4);

        // Normales
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

        // --- ILUMINACIÓN DINÁMICA ---
        float diff = fmaxf(dot(normal, p.sunDir), 0.0f);
        
        // Sombra proyectada
        float shadowVal = 1.0f;
        // Solo calculamos sombras si la cara mira al sol (para ahorrar cálculos) y no es agua
        if (diff > 0.0f && !isWater) {
            shadowVal = calculateShadow({worldX, finalY, worldZ}, p.sunDir, p);
        }

        float ambient = 0.15f; 
        float lightIntensity = clamp(ambient + diff * shadowVal, 0.0f, 1.0f);

        // Colores base (Biomas)
        float r, g, b;
        bool canHaveTree = false;

        if (isWater) {
            // Agua refleja el color del cielo + un tinte azul
            r = p.skyColor.x * 0.5f; 
            g = p.skyColor.y * 0.5f + 0.1f; 
            b = p.skyColor.z * 0.8f + 0.2f;
            lightIntensity = 0.8f + diff * 0.4f; // Specular simple
        } else {
            float hRel = clamp((finalY - p.waterLevel) / (p.heightMult - p.waterLevel), 0.0f, 1.0f);
            if (hRel < 0.05f) { r=0.8f; g=0.75f; b=0.5f; } // Arena
            else if (hRel > 0.75f) { r=0.95f; g=0.95f; b=1.0f; } // Nieve
            else {
                if (humidity < 0.45f) { r=0.6f; g=0.5f; b=0.3f; } // Árido
                else { r=0.1f; g=0.5f; b=0.1f; canHaveTree = true; } // Bosque
            }
        }

        // Tinte del atardecer: Si el sol está bajo, todo se pone rojizo
        if (p.sunDir.y < 0.2f && p.sunDir.y > -0.2f) {
            float sunsetFactor = 1.0f - abs(p.sunDir.y) * 5.0f;
            r += sunsetFactor * 0.3f;
            g -= sunsetFactor * 0.1f;
        }

        r *= lightIntensity; g *= lightIntensity; b *= lightIntensity;

        // Niebla (Usa el color dinámico del cielo)
        float dist = sqrtf(cx*cx + cy*cy);
        float fogFactor = clamp((dist - 0.75f) / 0.5f, 0.0f, 1.0f);
        r = lerp(r, p.skyColor.x, fogFactor);
        g = lerp(g, p.skyColor.y, fogFactor);
        b = lerp(b, p.skyColor.z, fogFactor);

        vertices[idx] = { cx, finalY, cy, 1.0f, r, g, b, 1.0f };

        // Árboles
        float treeProb = hash(worldX * 60.0f, worldZ * 60.0f); 
        if (canHaveTree && treeProb > 0.985f) {
            float tr = 0.0f; float tg = 0.2f * lightIntensity; float tb = 0.0f;
            tr = lerp(tr, p.skyColor.x, fogFactor); 
            tg = lerp(tg, p.skyColor.y, fogFactor); 
            tb = lerp(tb, p.skyColor.z, fogFactor);
            vertices[treeIdx] = { cx, finalY + 0.06f, cy, 1.0f, tr, tg, tb, 1.0f };
        } else {
            vertices[treeIdx] = { 0,0,0,0, 0,0,0,0.0f };
        }
    }
}

// Host functions
void initCudaMemory() { size_t s = TOTAL_VERTICES * sizeof(Vertex); cudaMalloc((void**)&d_vertices, s); }
void cleanupCudaMemory() { if (d_vertices) cudaFree(d_vertices); }
void runCudaKernel(Vertex* host_ptr, TerrainParams params) {
    if (!d_vertices) return;
    dim3 b(16, 16);
    dim3 g((MESH_WIDTH + b.x - 1) / b.x, (MESH_HEIGHT + b.y - 1) / b.y);
    terrain_kernel<<<g, b>>>(d_vertices, MESH_WIDTH, MESH_HEIGHT, params);
    cudaMemcpy(host_ptr, d_vertices, TOTAL_VERTICES * sizeof(Vertex), cudaMemcpyDeviceToHost);
}