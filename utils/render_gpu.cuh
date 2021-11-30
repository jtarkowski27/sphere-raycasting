#ifndef RENDER_GPU_H
#define RENDER_GPU_H

#include <helper_math.h>

__host__ __device__ void make_pixel(int pixel_index, int r, int g, int b, unsigned char *pixels);

__global__ void render_gpu(unsigned char *pixels, int max_x, int max_y, s_scene scene);

__host__ __device__ float3 color(s_ray &r, const s_scene scene);

__host__ __device__ void get_ray(s_camera camera, float u, float v, s_ray &r);

__host__ __device__ float3 calc_phong_reflection(s_lights &lights, s_hit_record &rec);

#endif