#ifndef LIGHT_H
#define LIGHT_H

#include "../objects/object.cuh"

struct s_lights
{
    int n;

    s_positions lpos;

    s_colors im;
    s_colors id;
    s_colors is;
};

#endif