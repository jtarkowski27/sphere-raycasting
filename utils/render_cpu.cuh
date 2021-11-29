#ifndef RENDER_CPU_H
#define RENDER_CPU_H

#include "../scene/scene.cuh"
#include "../scene/light.cuh"
#include "../scene/camera.cuh"

#include "../objects/object.cuh"
#include "../objects/sphere.cuh"

#include "render_gpu.cuh"

#include <float.h>
#include <helper_math.h>

__host__ void render_cpu(unsigned char *pixels, int max_x, int max_y, s_scene& scene);

#endif