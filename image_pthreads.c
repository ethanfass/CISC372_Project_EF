#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    const unsigned char* in; // input image (read-only)
    unsigned char* out;      // output image (thread writes its rows)
    int w, h, c;             // width, height, channels
    const float* k;          // 3x3 kernel (row-major, length 9)
    float div;               // divisor/normalizer
    float bias;              // bias to add
    int y0, y1;              // [y0, y1) rows for this thread
} Task;

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void* worker(void* arg) {
    Task* T = (Task*)arg;
    const int w = T->w, h = T->h, c = T->c;
    const unsigned char* in = T->in;
    unsigned char* out = T->out;
    const float* K = T->k;
    const float div = T->div;
    const float bias = T->bias;

    // Convolve per channel; handle edges by clamping coordinates
    for (int y = T->y0; y < T->y1; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                float acc = 0.0f;
                int idx = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    int yy = clampi(y + ky, 0, h - 1);
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xx = clampi(x + kx, 0, w - 1);
                        int src = (yy * w + xx) * c + ch;
                        acc += K[idx++] * (float)in[src];
                    }
                }
                float v = acc / div + bias;
                int dst = (y * w + x) * c + ch;
                out[dst] = (unsigned char)clampi((int)lroundf(v), 0, 255);
            }
        }
    }
    return NULL;
}

static void pick_kernel(const char* name, float K[9], float* div, float* bias) {
    // Default identity
    float identity[9] = {
         0, 0, 0,
         0, 1, 0,
         0, 0, 0
    };
    memcpy(K, identity, sizeof(identity));
    *div = 1.0f; *bias = 0.0f;

    if (!name) return;

    if (strcmp(name, "edge") == 0 || strcmp(name, "edge detection") == 0) {
        float k[9] = {
            -1,-1,-1,
            -1, 8,-1,
            -1,-1,-1
        };
        memcpy(K, k, sizeof(k)); *div = 1.0f; *bias = 0.0f;
    } else if (strcmp(name, "sharpen") == 0) {
        float k[9] = {
             0,-1, 0,
            -1, 5,-1,
             0,-1, 0
        };
        memcpy(K, k, sizeof(k)); *div = 1.0f; *bias = 0.0f;
    } else if (strcmp(name, "blur") == 0) {
        float k[9] = {
            1,1,1,
            1,1,1,
            1,1,1
        };
        memcpy(K, k, sizeof(k)); *div = 9.0f; *bias = 0.0f;
    } else if (strcmp(name, "gaussian") == 0) {
        float k[9] = {
            1,2,1,
            2,4,2,
            1,2,1
        };
        memcpy(K, k, sizeof(k)); *div = 16.0f; *bias = 0.0f;
    } else if (strcmp(name, "emboss") == 0) {
        float k[9] = {
            -2,-1, 0,
            -1, 1, 1,
             0, 1, 2
        };
        memcpy(K, k, sizeof(k)); *div = 1.0f; *bias = 128.0f; // add bias to keep in range
    } else if (strcmp(name, "identity") == 0) {
        // already set
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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
    if (!input) {
        fprintf(stderr, "Failed to load %s\n", inPath);
        return 1;
    }
    unsigned char* output = (unsigned char*)malloc((size_t)w * h * c);
    if (!output) {
        fprintf(stderr, "Out of memory\n");
        stbi_image_free(input);
        return 1;
    }

    float K[9]; float div = 1.0f; float bias = 0.0f;
    pick_kernel(filter, K, &div, &bias);

    // Launch threads
    if (T > h) T = h; // no point having more threads than rows
    pthread_t* th = (pthread_t*)malloc(sizeof(pthread_t) * T);
    Task* args = (Task*)malloc(sizeof(Task) * T);

    double t0 = now_sec();

    int rowsPer = (h + T - 1) / T;
    for (int t = 0; t < T; ++t) {
        int y0 = t * rowsPer;
        int y1 = y0 + rowsPer;
        if (y1 > h) y1 = h;

        args[t].in = input;
        args[t].out = output;
        args[t].w = w; args[t].h = h; args[t].c = c;
        args[t].k = K;
        args[t].div = div;
        args[t].bias = bias;
        args[t].y0 = y0; args[t].y1 = y1;

        pthread_create(&th[t], NULL, worker, &args[t]);
    }

    for (int t = 0; t < T; ++t) {
        pthread_join(th[t], NULL);
    }

    double t1 = now_sec();
    printf("pthreads: %d threads, %.3f seconds\n", T, t1 - t0);

    // Write PNG (stride = w*c bytes per row)
    if (!stbi_write_png("output.png", w, h, c, output, w * c)) {
        fprintf(stderr, "Failed to write output.png\n");
    } else {
        printf("Wrote output.png\n");
    }

    free(th);
    free(args);
    free(output);
    stbi_image_free(input);
    return 0;
}
