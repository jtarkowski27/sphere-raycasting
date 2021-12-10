NVCC=nvcc

GLUT_LIBS=-lGL -lGLU -lglut
CU_LIBS=-I"/usr/local/cuda/samples/common/inc" 

LDLIBS=${GLUT_LIBS} ${CU_LIBS}

# NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -rdc=true
GENCODE_FLAGS  = -arch=sm_75

SRCS = main.cu
INCS = utils/render_gpu.cuh utils/render_cpu.cuh scene/scene.cuh scene/camera.cuh scene/light.cuh objects/object.cuh objects/sphere.cuh

all: clean cudarc

cudarc: cudarc.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudarc cudarc.o ${LDLIBS}

cudarc.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudarc.o -c main.cu ${LDLIBS}

clean:
	rm -f raycasting_cpu raycasting *.o
