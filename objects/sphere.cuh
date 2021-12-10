#ifndef SPHERE_H
#define SPHERE_H

#include "object.cuh"
#include <helper_math.h>

struct s_spheres
{
    int n;

    s_positions pos;
    s_colors color;

    float *radius;
    
    float *ka;
    float *kd;
    float *ks;

    float *alpha;
};

struct s_sphere
{
    float3 position;
    float r;
};

// Calculates point of intersection between line and sphere of index i
__host__ __device__ bool hit(s_ray &r, s_spheres spheres, int i, float t_min, float t_max, s_hit_record &rec)
{
    float radius = spheres.radius[i];

    float3 center = make_float3(spheres.pos.x[i], spheres.pos.y[i], spheres.pos.z[i]);
    float3 oc = r.origin - center;

    float a = dot(r.direction, r.direction);
    float b = dot(oc, r.direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;

    rec.viewer = -r.direction;
    rec.i = i;

    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.position = point_at_parameter(r, rec.t);
            rec.normal = (rec.position - center) / radius;
            // rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.position = point_at_parameter(r, rec.t);
            rec.normal = (rec.position - center) / radius;
            // rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

// Returns data about closest point of intersection between spheres and one ray
__host__ __device__ bool hit(s_ray &r, s_spheres spheres, float t_min, float t_max, s_hit_record &rec)
{
    s_hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < spheres.n; i++)
    {
        if (hit(r, spheres, i, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif