#define RENDER_GPU

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include <vector_types.h>
#include <helper_math.h>
#include <helper_timer.h>

#include "scene/scene.cuh"

#include "utils/render_gpu.cuh"
#include "utils/render_cpu.cuh"

// OpenGL Graphics includes
#ifndef OPENGL_HEADERS
#define OPENGL_HEADERS
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif
#endif

int WIDTH = 1600;
int HEIGHT = 900;

bool GPU_RENDER_ENABLED = true;

GLubyte *h_bitmap;
GLubyte *d_bitmap;

int tx = 16;
int ty = 16;

int nx = WIDTH;
int ny = HEIGHT;

int num_pixels = nx * ny;
size_t bitmap_size = num_pixels * sizeof(GLubyte);

int SPHERES_COUNT = 1000;
int LIGHTS_COUNT = 200;

s_scene d_scene;
s_scene h_scene;

int resolution_horizontal = WIDTH;
int resolution_vertical = HEIGHT;
float fov;

StopWatchInterface *fps_timer = NULL;

float angle_x = 0;
float angle_y = 0;
int state = 1;

clock_t start, stop;
clock_t second_start, second_stop;
double raycasting_time = 0;
double cpu_to_gpu_copying_time = 0;
double gpu_to_cpu_copying_time = 0;

int start_x = -1;
int start_y = -1;

float start_angle_x = 0;
float start_angle_y = 0;

float prev_angle_x = 0;
float prev_angle_y = 0;

int shift_pressed = 0;

int frames;
int fpsCount = 0;
float avgFPS = 1.0f;
int fpsLimit = 1;
double first_second = 2;

char fps[512];

// Return radom float from range
float rand_float(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}

// glutDisplayFunc() event handler
void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, h_bitmap);
    glutSwapBuffers();
}

// Generate one random 3D point and assign its coordinates at index i
void assign_position(s_positions &positions, int i)
{
    float x = rand_float(-150, 150);
    float y = rand_float(-150, 150);

    float dist = sqrt(x * x + y * y);

    float s_x = x / dist;
    float s_y = y / dist;

    positions.angle[i] = atan2(s_y, s_x);

    positions.x[i] = x;
    positions.y[i] = y;
    positions.z[i] = rand_float(-100, 100);
}

// Generate random parameters of spheres and lights
void randomize_scene_variables()
{
    d_scene.camera.origin = make_float3(0, 100, 100);

    set_resolution(d_scene.camera, WIDTH, HEIGHT);
    look_at(d_scene.camera, 0, 0, 0);

    h_scene.camera.origin = make_float3(-1000, 600, 700);

    set_resolution(h_scene.camera, WIDTH, HEIGHT);
    look_at(h_scene.camera, 0, 0, 0);

    for (int i = 0; i < SPHERES_COUNT; i++)
    {
        h_scene.spheres.radius[i] = rand_float(2, 4);
        h_scene.spheres.ka[i] = rand_float(0, 0.2);
        h_scene.spheres.kd[i] = rand_float(0, 0.03);
        h_scene.spheres.ks[i] = rand_float(0, 0.4);
        h_scene.spheres.alpha[i] = (rand() % 400) + 10;

        assign_position(h_scene.spheres.pos, i);

        h_scene.spheres.color.r[i] = rand_float(0, 1);
        h_scene.spheres.color.g[i] = rand_float(0, 1);
        h_scene.spheres.color.b[i] = rand_float(0, 1);
    }

    for (int i = 0; i < LIGHTS_COUNT; i++)
    {
        assign_position(h_scene.lights.lpos, i);

        h_scene.lights.im.r[i] = rand_float(0, 1);
        h_scene.lights.im.g[i] = rand_float(0, 1);
        h_scene.lights.im.b[i] = rand_float(0, 1);

        h_scene.lights.id.r[i] = rand_float(0, 1);
        h_scene.lights.id.g[i] = rand_float(0, 1);
        h_scene.lights.id.b[i] = rand_float(0, 1);

        h_scene.lights.is.r[i] = rand_float(0, 1);
        h_scene.lights.is.g[i] = rand_float(0, 1);
        h_scene.lights.is.b[i] = rand_float(0, 1);
    }
}

// Rotate list of positions
void rotate_objects(s_positions *positions, int n, float rotate)
{
    for (int i = 0; i < n; i++)
    {
        float x = positions->x[i];
        float y = positions->y[i];

        float angle = positions->angle[i] + rotate;
        float dist = sqrt(x * x + y * y);

        positions->x[i] = dist * cos(angle);
        positions->y[i] = dist * sin(angle);

        positions->angle[i] = angle;
    }
}

// Memcpy GPU->CPU
void memcpy_device_to_host()
{
    checkCudaErrors(cudaMemcpy(h_bitmap, d_bitmap, bitmap_size, cudaMemcpyDeviceToHost));
}

// Memcpy CPU->GPU
void memcpy_host_to_device()
{
    checkCudaErrors(cudaMemcpy(d_scene.spheres.radius, h_scene.spheres.radius, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_scene.spheres.ka, h_scene.spheres.ka, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.spheres.kd, h_scene.spheres.kd, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.spheres.ks, h_scene.spheres.ks, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.spheres.alpha, h_scene.spheres.alpha, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_scene.spheres.pos.x, h_scene.spheres.pos.x, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.spheres.pos.y, h_scene.spheres.pos.y, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.spheres.pos.z, h_scene.spheres.pos.z, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_scene.spheres.color.r, h_scene.spheres.color.r, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.spheres.color.g, h_scene.spheres.color.g, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.spheres.color.b, h_scene.spheres.color.b, sizeof(float) * SPHERES_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_scene.lights.lpos.x, h_scene.lights.lpos.x, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.lpos.y, h_scene.lights.lpos.y, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.lpos.z, h_scene.lights.lpos.z, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_scene.lights.im.r, h_scene.lights.im.r, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.im.g, h_scene.lights.im.g, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.im.b, h_scene.lights.im.b, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_scene.lights.id.r, h_scene.lights.id.r, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.id.g, h_scene.lights.id.g, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.id.b, h_scene.lights.id.b, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_scene.lights.is.r, h_scene.lights.is.r, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.is.g, h_scene.lights.is.g, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scene.lights.is.b, h_scene.lights.is.b, sizeof(float) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
}

// Allocate memory for bitmap
void malloc_bitmap()
{
    num_pixels = nx * ny;
    bitmap_size = num_pixels * sizeof(GLubyte) * 3;
    free(h_bitmap);
    h_bitmap = new GLubyte[nx * ny * 3];

    cudaFree(d_bitmap);
    checkCudaErrors(cudaMallocManaged((void **)&d_bitmap, bitmap_size));
}

// Allocate memory for scene
void malloc_scene()
{
    sdkCreateTimer(&fps_timer);

    h_scene.spheres.pos.x = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.pos.y = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.pos.z = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.pos.angle = (float *)malloc(sizeof(float) * SPHERES_COUNT);

    h_scene.spheres.color.r = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.color.g = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.color.b = (float *)malloc(sizeof(float) * SPHERES_COUNT);

    h_scene.spheres.radius = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.ka = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.kd = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.ks = (float *)malloc(sizeof(float) * SPHERES_COUNT);
    h_scene.spheres.alpha = (float *)malloc(sizeof(float) * SPHERES_COUNT);

    h_scene.lights.lpos.x = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.lpos.y = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.lpos.z = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    
    h_scene.lights.im.r = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.im.g = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.im.b = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    
    h_scene.lights.is.r = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.is.g = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.is.b = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    
    h_scene.lights.id.r = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.id.g = (float *)malloc(sizeof(float) * LIGHTS_COUNT);
    h_scene.lights.id.b = (float *)malloc(sizeof(float) * LIGHTS_COUNT);

    h_scene.lights.lpos.angle = (float *)malloc(sizeof(float) * LIGHTS_COUNT);

    malloc_bitmap();

    d_scene.spheres.n = SPHERES_COUNT;
    h_scene.spheres.n = SPHERES_COUNT;

    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.pos.x, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.pos.y, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.pos.z, SPHERES_COUNT * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.color.r, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.color.g, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.color.b, SPHERES_COUNT * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.radius, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.ka, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.kd, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.ks, SPHERES_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.spheres.alpha, SPHERES_COUNT * sizeof(float)));

    d_scene.lights.n = LIGHTS_COUNT;
    h_scene.lights.n = LIGHTS_COUNT;

    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.lpos.x, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.lpos.y, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.lpos.z, LIGHTS_COUNT * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.id.r, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.id.g, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.id.b, LIGHTS_COUNT * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.im.r, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.im.g, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.im.b, LIGHTS_COUNT * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.is.r, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.is.g, LIGHTS_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_scene.lights.is.b, LIGHTS_COUNT * sizeof(float)));

}

// Free allocated space
void free_scene()
{
    sdkDeleteTimer(&fps_timer);

    free(h_scene.spheres.pos.x);
    free(h_scene.spheres.pos.y);
    free(h_scene.spheres.pos.z);
    free(h_scene.spheres.pos.angle);

    free(h_scene.spheres.color.r);
    free(h_scene.spheres.color.g);
    free(h_scene.spheres.color.b);

    free(h_scene.spheres.radius);
    free(h_scene.spheres.ka);
    free(h_scene.spheres.kd);
    free(h_scene.spheres.ks);
    free(h_scene.spheres.alpha);

    free(h_scene.lights.lpos.x);
    free(h_scene.lights.lpos.y);
    free(h_scene.lights.lpos.z);
    
    free(h_scene.lights.im.r);
    free(h_scene.lights.im.g);
    free(h_scene.lights.im.b);
    
    free(h_scene.lights.id.r);
    free(h_scene.lights.id.g);
    free(h_scene.lights.id.b);
    
    free(h_scene.lights.is.r);
    free(h_scene.lights.is.g);
    free(h_scene.lights.is.b);

    free(h_scene.lights.lpos.angle);

    cudaFree(d_bitmap);

    cudaFree(d_scene.spheres.pos.x);
    cudaFree(d_scene.spheres.pos.y);
    cudaFree(d_scene.spheres.pos.z);

    cudaFree(d_scene.spheres.color.r);
    cudaFree(d_scene.spheres.color.g);
    cudaFree(d_scene.spheres.color.b);

    cudaFree(d_scene.spheres.radius);
    cudaFree(d_scene.spheres.ka);
    cudaFree(d_scene.spheres.kd);
    cudaFree(d_scene.spheres.ks);
    cudaFree(d_scene.spheres.alpha);

    cudaFree(d_scene.lights.lpos.x);
    cudaFree(d_scene.lights.lpos.y);
    cudaFree(d_scene.lights.lpos.z);

    cudaFree(d_scene.lights.im.r);
    cudaFree(d_scene.lights.im.g);
    cudaFree(d_scene.lights.im.b);

    cudaFree(d_scene.lights.id.r);
    cudaFree(d_scene.lights.id.g);
    cudaFree(d_scene.lights.id.b);

    cudaFree(d_scene.lights.is.r);
    cudaFree(d_scene.lights.is.g);
    cudaFree(d_scene.lights.is.b);

    free(h_bitmap);
}

// TODO event of window resizing
void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, h, 0);
    glMatrixMode(GL_MODELVIEW);
}

// Execute Raycasting algorithm and read-write calculated values
void render_scene()
{
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    start = clock();
    if (GPU_RENDER_ENABLED)
    {
        render_gpu<<<blocks, threads>>>(d_bitmap, nx, ny, d_scene);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        render_cpu(h_bitmap, nx, ny, h_scene);
    }

    stop = clock();
    raycasting_time = ((double)(stop - start)) / CLOCKS_PER_SEC;

    start = clock();
    if (GPU_RENDER_ENABLED)
    {
        memcpy_device_to_host();
    }

    stop = clock();
    gpu_to_cpu_copying_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
}

// Display measured times of algorithm execution
void computeFPS()
{
    frames++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&fps_timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&fps_timer);
    }

    second_stop = clock();
    double second_passed = ((double)(second_stop - second_start)) / CLOCKS_PER_SEC + first_second;

    if (second_passed > 1)
    {
        first_second = 0;
        second_start = clock();

        if (GPU_RENDER_ENABLED)
        {
            sprintf(fps, "Spheres Raycasting: %.2f fps (Raycasting: %.4f s, CPU->GPU copying: %.6f s, GPU->CPU copying: %.6f s)",
                    avgFPS, raycasting_time, cpu_to_gpu_copying_time, gpu_to_cpu_copying_time);
        }
        else
        {
            sprintf(fps, "Spheres Raycasting: %.2f fps (Raycasting: %.4f s)",
                    avgFPS, raycasting_time);
        }

        std::cout << fps << "\n";
    }

    glutSetWindowTitle(fps);
}

// glutTimerFunc() event handler
void timer(int)
{
    sdkStartTimer(&fps_timer);
    glutPostRedisplay();

    glutTimerFunc(1000 / 100, timer, 0);

    float angle_diff = -(angle_x - prev_angle_x);

    if (shift_pressed)
    {
        rotate_objects(&(h_scene.lights.lpos), LIGHTS_COUNT, angle_diff);
    }
    else
    {
        rotate_objects(&(h_scene.spheres.pos), SPHERES_COUNT, angle_diff);
    }

    prev_angle_x = angle_x;
    prev_angle_y = angle_y;

    if (GPU_RENDER_ENABLED)
    {
        start = clock();
        memcpy_host_to_device();
        stop = clock();
        cpu_to_gpu_copying_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    }

    render_scene();

    sdkStopTimer(&fps_timer);
    computeFPS();
}

// glutMotionFunc() event handler
void mouse(int button, int state, int x, int y)
{
    shift_pressed = glutGetModifiers() & GLUT_ACTIVE_SHIFT;

    if (state == GLUT_DOWN)
    {
        start_x = x;
        start_y = y;

        start_angle_x = angle_x;
        start_angle_y = angle_y;
    }
}

// glutMotionFunc() event handler
void drag(int x, int y)
{
    angle_x = start_angle_x + ((float)(x - start_x) / 300.0);
    angle_y = start_angle_y + ((float)(y - start_y) / 300.0);
}

// Setup important event handlers for OpenGL
void setup_opengl(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GL_DOUBLE);

    glutInitWindowSize(nx, ny);
    glutInitWindowPosition(100, 100);

    int MainWindow = glutCreateWindow("Sphere Raycasting");
    glClearColor(0.0, 0.0, 0.0, 0);

    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutReshapeFunc(reshape);
    glutMotionFunc(drag);
    glutTimerFunc(0, timer, 0);
    glutMainLoop();
}

// Parsing command-line arguments
void main_parameters(int argc, char *argv[])
{
    char spheres_str[] = "-s\0";
    char lights_str[] = "-l\0";
    char cpu_str[] = "-cpu\0";

    int lights_count = -1;
    int spheres_count = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], cpu_str, sizeof(cpu_str)) == 0)
        {
            GPU_RENDER_ENABLED = false;

            LIGHTS_COUNT = min(10, LIGHTS_COUNT);
            SPHERES_COUNT = min(100, SPHERES_COUNT);
        }
        if (strncmp(argv[i], lights_str, sizeof(lights_str)) == 0 && i < argc - 1)
        {
            lights_count = atoi(argv[i + 1]);
        }
        if (strncmp(argv[i], spheres_str, sizeof(spheres_str)) == 0 && i < argc - 1)
        {
            spheres_count = atoi(argv[i + 1]);
        }
    }

    if (lights_count > -1)
    {
        LIGHTS_COUNT = lights_count;
    }

    if (spheres_count > -1)
    {
        SPHERES_COUNT = spheres_count;
    }

    nx = WIDTH;
    ny = HEIGHT;

    num_pixels = nx * ny;
    bitmap_size = num_pixels * sizeof(GLubyte);

    resolution_horizontal = WIDTH;
    resolution_vertical = HEIGHT;
}

int main(int argc, char *argv[])
{
    main_parameters(argc, argv);

    std::cerr << "Raycasting a scene of " << SPHERES_COUNT << " spheres and " << LIGHTS_COUNT << " lights with " << nx << "x" << ny;
    std::cerr << " rays in " << tx << "x" << ty << " blocks.\n";

    second_start = clock();

    malloc_scene();
    checkCudaErrors(cudaGetLastError());
    randomize_scene_variables();
    memcpy_host_to_device();

    render_scene();

    setup_opengl(0, argv);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_scene();

    cudaDeviceReset();
}
