#include "object.cuh"

// Returns coordinates of point translated by direction vector of ray
__host__ __device__ float3 point_at_parameter(s_ray &ray, float t)
{
    return ray.origin + t * ray.direction;
}