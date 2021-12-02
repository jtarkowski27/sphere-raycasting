CC=nvcc

GLUT_LIBS=-lGL -lGLU -lglut
CU_LIBS=-I"/usr/local/cuda/samples/common/inc" -I"/opt/cuda/samples/common/inc"

LDLIBS=${GLUT_LIBS} ${CU_LIBS}

FILES=scene.o light.o camera.o object.o sphere.o render_gpu.o render_cpu.o 


all: raycasting 

raycasting: main.o  ${FILES}
	${CC} main.o ${FILES} -o raycasting ${LDLIBS}

main.o:	main.cu
	${CC} -dc -c main.cu -o main.o ${LDLIBS}

# raycasting_cpu: main_cpu.o  ${FILES}
# 	${CC} main_cpu.o  ${FILES} -o raycasting_cpu ${LDLIBS}

# main_cpu.o:	main_cpu.cu
# 	${CC} -dc -c main_cpu.cu -o main_cpu.o ${LDLIBS}

render_gpu.o: utils/render_gpu.cu
	${CC} -dc -c utils/render_gpu.cu -o render_gpu.o ${LDLIBS}

render_cpu.o: utils/render_cpu.cu
	${CC} -dc -c utils/render_cpu.cu -o render_cpu.o ${LDLIBS}

scene.o: scene/scene.cu
	${CC} -dc -c scene/scene.cu -o scene.o ${LDLIBS} 

camera.o: scene/camera.cu
	${CC} -dc  -c scene/camera.cu -o camera.o ${LDLIBS} 

light.o: scene/light.cu
	${CC} -dc -c scene/light.cu -o light.o ${LDLIBS} 


object.o: objects/object.cu
	${CC} -dc -c objects/object.cu -o object.o ${LDLIBS} 

sphere.o: objects/sphere.cu
	${CC} -dc -c objects/sphere.cu -o sphere.o ${LDLIBS} 

clean:
	rm -f raycasting_cpu raycasting *.o
