#include "cpu_kernel.h"
#include <math.h>
#include <algorithm>

// --- MATEMÁTICAS (Versión CPU estandar) ---
float fract(float x) { return x - floorf(x); }

float hash(float x, float y) { 
    return fract(sinf(x * 12.9898f + y * 78.233f) * 43758.5453123f); 
}

float lerp(float a, float b, float t) { return a + t * (b - a); }

float clamp(float v, float minVal, float maxVal) { 
    return std::max(minVal, std::min(v, maxVal)); 
}

float noise(float x, float y) {
    float ix = floorf(x); float iy = floorf(y);
    float fx = x - ix; float fy = y - iy;
    float a = hash(ix, iy); float b = hash(ix + 1.0f, iy);
    float c = hash(ix, iy + 1.0f); float d = hash(ix + 1.0f, iy + 1.0f);
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    return lerp(lerp(a, b, ux), lerp(c, d, ux), uy);
}

float fbm(float x, float y, int octaves) {
    float total = 0.0f; float amp = 0.5f; float freq = 1.0f; float maxV = 0.0f;
    for(int i=0; i<octaves; i++) { 
        total += noise(x*freq, y*freq)*amp; 
        maxV+=amp; amp*=0.5f; freq*=2.0f; 
    }
    return total/maxV;
}

float ridge_noise(float x, float y, int octaves) {
    float total = 0.0f; float amp = 1.0f; float freq = 1.0f; float maxV = 0.0f;
    for(int i=0; i<octaves; i++) { 
        float n = 1.0f - fabs(noise(x*freq, y*freq)); 
        n*=n; total+=n*amp; maxV+=amp; amp*=0.5f; freq*=2.0f; 
    }
    return total/maxV;
}

void normalize(float& x, float& y, float& z) {
    float len = sqrtf(x*x + y*y + z*z);
    if(len > 0.0f){ x/=len; y/=len; z/=len; }
}

float dot(float x1, float y1, float z1, float x2, float y2, float z2) { 
    return x1*x2 + y1*y2 + z1*z2; 
}

// --- CÁLCULO DE SOMBRAS PROYECTADAS (CPU) ---
float calculateShadow(float px, float py, float pz, float sunX, float sunY, float sunZ, TerrainParams p) {
    if (p.enableShadows == 0) return 1.0f;

    float shadow = 1.0f;
    float stepSize = 0.08f * p.scale;
    
    float cX = px; float cY = py; float cZ = pz;
    
    // Bucle pesado para la CPU
    for (int i = 1; i <= 16; i++) {
        cX += sunX * stepSize;
        cY += sunY * stepSize;
        cZ += sunZ * stepSize;

        if (cY > p.heightMult) break;

        float h = ridge_noise(cX, cZ, 2);
        float groundY = h * p.heightMult - 0.5f;

        if (cY < groundY) {
            shadow = 0.2f;
            break;
        }
    }
    return shadow;
}

// --- FUNCIÓN PRINCIPAL CPU ---
void generateTerrainCPU(std::vector<Vertex>& vertices, int width, int height, TerrainParams p) {
    // BUCLE SECUENCIAL: Aquí es donde la CPU sufre vs la GPU
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
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
            
            float nx = finalY - h_r;
            float ny = eps;
            float nz = finalY - h_d;
            normalize(nx, ny, nz);

            bool isWater = finalY < p.waterLevel;
            if (isWater) {
                finalY = p.waterLevel;
                nx = 0.0f; ny = 1.0f; nz = 0.0f;
            }

            float diff = std::max(dot(nx, ny, nz, p.sunDir.x, p.sunDir.y, p.sunDir.z), 0.0f);
            
            float shadowVal = 1.0f;
            if (diff > 0.0f && !isWater) {
                shadowVal = calculateShadow(worldX, finalY, worldZ, p.sunDir.x, p.sunDir.y, p.sunDir.z, p);
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
                
                // 1. PLAYA
                if (hRel < 0.05f) { 
                    r=0.8f; g=0.75f; b=0.5f; 
                }
                // 2. NIEVE
                else if (hRel > 0.85f) { 
                    r=0.95f; g=0.95f; b=1.0f; 
                }
                // 3. ROCA
                else if (hRel > 0.55f) {
                    r=0.35f; g=0.35f; b=0.38f; 
                    float noiseRock = hash(worldX * 10.0f, worldZ * 10.0f); 
                    r += noiseRock * 0.05f; g += noiseRock * 0.05f; b += noiseRock * 0.05f;
                }
                // 4. VEGETACIÓN
                else {
                    if (humidity < 0.45f) { r=0.6f; g=0.5f; b=0.3f; }
                    else { r=0.1f; g=0.5f; b=0.1f; }
                }
            }

            if (p.sunDir.y < 0.2f && p.sunDir.y > -0.2f) {
                float sunsetFactor = 1.0f - std::abs(p.sunDir.y) * 5.0f;
                r += sunsetFactor * 0.3f;
                g -= sunsetFactor * 0.1f;
            }

            r *= lightIntensity; g *= lightIntensity; b *= lightIntensity;

            float dist = sqrtf(cx*cx + cy*cy);
            float fogFactor = clamp((dist - 0.75f) / 0.5f, 0.0f, 1.0f);
            
            // Lerp manual para niebla
            r = r + fogFactor * (p.skyColor.x - r);
            g = g + fogFactor * (p.skyColor.y - g);
            b = b + fogFactor * (p.skyColor.z - b);

            vertices[idx] = { cx, finalY, cy, 1.0f, r, g, b, 1.0f };
        }
    }
}