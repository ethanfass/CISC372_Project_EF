#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void pick_kernel(const char* name, float K[9], float* div, float* bias) {
    float identity[9] = { 0,0,0, 0,1,0, 0,0,0 };
    memcpy(K, identity, sizeof(identity)); *div = 1.0f; *bias = 0.0f;
    if (!name) return;

    if (!strcmp(name, "edge") || !strcmp(name, "edge detection")) {
        float k[9] = { -1,-1,-1, -1,8,-1, -1,-1,-1 };
        memcpy(K, k, sizeof(k)); *div = 1.0f; *bias = 0.0f;
    } else if (!strcmp(name, "sharpen")) {
        float k[9] = { 0,-1,0, -1,5,-1, 0,-1,0 };
        memcpy(K, k, sizeof(k)); *div = 1.0f; *bias = 0.0f;
    } else if (!strcmp(name, "blur")) {
        float k[9] = { 1,1,1, 1,1,1, 1,1,1 };
        memcpy(K, k, sizeof(k)); *div = 9.0f; *bias = 0.0f;
    } else if (!strcmp(name, "gaussian")) {
        float k[9] = { 1,2,1, 2,4,2, 1,2,1 };
        memcpy(K, k, sizeof(k)); *div = 16.0f; *bias = 0.0f;
    } else if (!strcmp(name, "emboss")) {
        float k[9] = { -2,-1,0, -1,1,1, 0,1,2 };
        memcpy(K, k, sizeof(k)); *div = 1.0f; *bias = 128.0f;
    } // identity else already set
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <image.jpg/png> <filter> [threads]\n", argv[0]);
        fprintf(stderr, "Filters: edge|sharpen|blur|gaussian|emboss|identity\n");
        return 1;
    }
    const char* inPath = argv[1];
    const char* filter = argv[2];
    int T = (argc >= 4) ? atoi(argv[3]) : 4;
    if (T <= 0) T = 4;

    int w, h, c;
    unsigned char* input = stbi_load(inPath, &w, &h, &c, 0);
    if (!input) { fprintf(stderr, "Failed to load %s\n", inPath); return 1; }

    unsigned char* output = (unsigned char*)malloc((size_t)w * h * c);
    if (!output) { fprintf(stderr, "Out of memory\n"); stbi_image_free(input); return 1; }

    float K[9]; float div = 1.0f; float bias = 0.0f;
    pick_kernel(filter, K, &div, &bias);

    omp_set_num_threads(T);
    double t0 = omp_get_wtime();

    // Parallelize across image rows; each thread writes disjoint rows (no races).
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                float acc = 0.0f;
                int idx = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    int yy = clampi(y + ky, 0, h - 1);
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xx = clampi(x + kx, 0, w - 1);
                        int src = (yy * w + xx) * c + ch;
                        acc += K[idx++] * (float)input[src];
                    }
                }
                float v = acc / div + bias;
                int dst = (y * w + x) * c + ch;
                // clamp
                if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                output[dst] = (unsigned char)lroundf(v);
            }
        }
    }

    double t1 = omp_get_wtime();
    printf("openmp: %d threads, %.3f seconds\n", T, t1 - t0);

    if (!stbi_write_png("output.png", w, h, c, output, w * c)) {
        fprintf(stderr, "Failed to write output.png\n");
    } else {
        printf("Wrote output.png\n");
    }

    free(output);
    stbi_image_free(input);
    return 0;
}
