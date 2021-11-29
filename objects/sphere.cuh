#ifndef SPHERE_H
#define SPHERE_H

#include "object.cuh"
#include <helper_math.h>

struct s_spheres
{
    int n;

    s_positions pos;
    s_colors color;

    float *r;
};

struct s_sphere
{
    float3 position;
    float r;
};

__host__ __device__ bool hit(s_ray &r, s_spheres spheres, int i, float t_min, float t_max, s_hit_record &rec);

__host__ __device__ bool hit(s_ray &r, s_spheres spheres, float t_min, float t_max, s_hit_record &rec);

#endif