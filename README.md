# Sphere Raycasting

![Window screenshot](window.png)

## Description

This project contains the source code for basic 3D [ray caster](https://pl.wikipedia.org/wiki/Ray_casting) of spheres with [Phong reflection model](https://en.wikipedia.org/wiki/Phong_reflection_model).

Implementation was based on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) series and [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) from NVIDIA's Developer Blog.

## Dependencies

Following dependencies were used in this project:

* CUDA SDK 10.1
* OpenGL 4.6.0
* GNU Make 4.2.1

## Build and Run
### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
