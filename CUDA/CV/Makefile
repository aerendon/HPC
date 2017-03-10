CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main : main.cu
	nvcc $(CFLAGS) main.cu $(LIBS) -o main.out
