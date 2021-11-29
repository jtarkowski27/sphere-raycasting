#ifndef OBJECT_H
#define OBJECT_H

#include <helper_math.h>

struct s_hit_record
{
    float t;
    float3 position;
    float3 normal;
    float3 viewer;

    int i;
};

struct s_ray
{
    float3 origin;
    float3 direction;
};

struct s_colors
{
    int n;

    float *r;
    float *g;
    float *b;
};

struct s_positions
{
    int n;
    float *x;
    float *y;
    float *z;
    float *angle;
};

__host__ __device__ float3 point_at_parameter(s_ray &ray, float t);

#endif