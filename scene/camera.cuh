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

// Directs camera at certain point in 3d space
__device__ __host__ void look_at(s_camera &camera, float x, float y, float z)
{
    float3 a, b;

    camera.direction.x = x - camera.origin.x;
    camera.direction.y = y - camera.origin.y;
    camera.direction.z = z - camera.origin.z;

    normalize(camera.direction);

    if (abs(camera.direction.x) < EPS && abs(camera.direction.y) < EPS)
    {
        a = make_float3(0, -camera.direction.z, 0);
        b = make_float3(0, camera.direction.z, 0);
    }
    else
    {
        a = make_float3(camera.direction.y, -camera.direction.x, 0);
        b = make_float3(-camera.direction.y, camera.direction.x, 0);
    }

    camera.left = camera.aspect_ratio * normalize(a);
    camera.right = camera.aspect_ratio * normalize(b);

    camera.up = normalize(cross(camera.left, camera.direction));
    camera.down = normalize(cross(camera.right, camera.direction));

    camera.left_down = camera.origin + (camera.left + camera.down + 0.007 * camera.direction);
    camera.right_up = camera.origin + (camera.right + camera.up + 0.007 * camera.direction);

    camera.left_to_right = (camera.right - camera.left);
    camera.down_to_up = (camera.up - camera.down);
}

// Directs camera at certain point in 3d space
__device__ __host__ void look_at(s_camera &camera, float3 &point)
{
    look_at(camera, point.x, point.y, point.z);
}

// Sets resolution of camera
__device__ __host__ void set_resolution(s_camera &camera, int resolution_horizontal, int resolution_vertical)
{
    camera.resolution_horizontal = (float)resolution_horizontal;
    camera.resolution_vertical = (float)resolution_vertical;
    camera.aspect_ratio = (float)resolution_horizontal / (float)resolution_vertical;
}

#endif