#ifndef CAMERA_H
#define CAMERA_H


#ifndef EPS
#define EPS 0.0001
#endif

#include <helper_math.h>

struct s_camera
{
    float3 origin;
    float3 direction;

    float3 left;
    float3 right;

    float3 up;
    float3 down;

    float3 left_down;
    float3 right_up;

    float3 left_to_right;
    float3 down_to_up;

    float resolution_horizontal;
    float resolution_vertical;

    float aspect_ratio;

    float fov;
};


__device__ __host__ void look_at(s_camera &camera, float x, float y, float z);

__device__ __host__ void look_at(s_camera &camera, float3 &point);

__device__ __host__ void set_resolution(s_camera &camera, int resolution_horizontal, int resolution_vertical);

#endif