#ifndef SCENE_H
#define SCENE_H

#include "../objects/object.cuh"
#include "../objects/sphere.cuh"

#include "light.cuh"
#include "camera.cuh"

struct s_scene
{
    s_camera camera;
    s_spheres spheres;
    s_lights lights;
};


#endif
