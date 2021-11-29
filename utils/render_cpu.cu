#include "render_cpu.cuh"
#include "render_gpu.cuh"

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