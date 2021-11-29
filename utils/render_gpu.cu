#include "../scene/scene.cuh"
#include "../scene/light.cuh"
#include "../scene/camera.cuh"

#include "../objects/object.cuh"
#include "../objects/sphere.cuh"

#include "render_gpu.cuh"
#include <float.h>
#include <helper_math.h>

// Assigns the color of pixel to bitmap array
__host__ __device__ void make_pixel(int pixel_index, int r, int g, int b, unsigned char *pixels)
{
    int position = pixel_index * 3;

    pixels[position] = r;
    pixels[position + 1] = g;
    pixels[position + 2] = b;
}

// Calculates color to be displayed at the point of intersection
__host__ __device__ float3 calc_point_light(s_lights &lights, s_hit_record &rec, s_spheres &spheres)
{
    float3 ia = make_float3(spheres.color.r[rec.i], spheres.color.g[rec.i], spheres.color.b[rec.i]);
    float3 im = make_float3(1.0f, 1.0f, 1.0f);
    float3 id = ia;
    float3 is = make_float3(1.0f, 1.0f, 1.0f);

    float3 N = rec.normal;
    float3 V = rec.viewer;

    N = normalize(N);
    V = normalize(V);

    float3 diffuse = make_float3(0, 0, 0);
    float3 specular = make_float3(0, 0, 0);

    float ka = 0.1;
    float kd = 0.01;
    float ks = 0.9;
    float alpha = 10000;

    float3 ka_ia = ka * ia;
    float3 kd_im_id = kd * im * id;
    float3 ks_im_is = ks * im * is;

    for (int i = 0; i < lights.n; i++)
    {
        float3 position = make_float3(lights.pos.x[i], lights.pos.y[i], lights.pos.z[i]);
        float3 L = position - rec.position;

        L = normalize(L);

        float3 R = 2 * dot(L, N) * N - L;

        R = normalize(L);

        float dot_L_N = max(dot(L, N), 0.0);
        float dot_R_V = max(dot(R, V), 0.0);
        float dot_R_V_to_alpha = pow(dot_R_V, alpha);

        specular += ks_im_is * dot_R_V_to_alpha;
        diffuse += kd_im_id * dot_L_N;
    }

    float3 Ip = ka_ia + diffuse + specular;

    float r = max(min(Ip.x, 1.0f)*255, 0.0f);
    float g = max(min(Ip.y, 1.0f)*255, 0.0f);
    float b = max(min(Ip.z, 1.0f)*255, 0.0f);

    return make_float3(r, g, b);
}

// Returns either calculated color of black
__host__ __device__ float3 color(s_ray &r, s_scene scene)
{
    s_ray cur_ray = r;

    s_hit_record rec;
    if (hit(cur_ray, scene.spheres, 0.001f, FLT_MAX, rec))
    {
        return calc_point_light(scene.lights, rec, scene.spheres);
    }
    else
    {
        return {0.0f, 0.0f, 0.0f};
    }
}

// Main kernel, considers each ray parallelly
__global__ void render_gpu(unsigned char *pixels, int max_x, int max_y, s_scene scene)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int pixel_index = j * max_x + i;

    float u = float(i);
    float v = float(j);

    s_ray r;
    get_ray(scene.camera, u, v, r);

    float3 c = color(r, scene);

    make_pixel(pixel_index, (int)c.x, (int)c.y, (int)c.z, pixels);
}

// Returns ray shot from camera based on its index its index
__host__ __device__ void get_ray(s_camera camera, float u, float v, s_ray &r)
{
    float3 target = camera.left_down + camera.left_to_right * u / camera.resolution_horizontal + camera.down_to_up * v / camera.resolution_vertical;
    r.origin = camera.origin;
    r.direction = normalize(target - r.origin);
}