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

// Main kernel, considers each ray parallelly
__host__ void render_cpu(unsigned char *pixels, int max_x, int max_y, s_scene& scene)
{
    for (int i = 0; i < max_x; i++)
    {
        for (int j = 0; j < max_y; j++)
        {
            int pixel_index = j * max_x + i;

            float u = float(i);
            float v = float(j);

            s_ray r;
            get_ray(scene.camera, u, v, r);

            float3 c = color(r, scene);

            make_pixel(pixel_index, (int)c.x, (int)c.y, (int)c.z, pixels);
        }
    }
}

#endif