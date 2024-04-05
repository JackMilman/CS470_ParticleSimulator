objects = src/cuda/main.o
objects_NV = src_NV/cuda/main_nv.o
GL_LIBS = -L/usr/lib/x86_64-linux-gnu -lGLEW -lGL -lGLU -lglut

visual: $(objects)
	nvcc -gencode=arch=compute_52,code=\"sm_52,compute_52\" $(objects) -o app $(GL_LIBS)
	g++ -std=c++11 -I. -o app_serial src/serial/main_serial.cpp $(GL_LIBS)

non-visual: $(objects_NV)
	nvcc -gencode=arch=compute_52,code=\"sm_52,compute_52\" $(objects_NV) -o app_nv
	g++ -std=c++11 -I. -o app_serial_nv src/serial/main_serial_nv.cpp

%.o: %.cpp
	nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc $(GL_LIBS) $< -o $@
		
%.o: %.cu
	nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc $(GL_LIBS) $< -o $@

clean:
	find . -name "*.o" -delete
	find . -name "app" -delete
	find . -name "app_serial" -delete