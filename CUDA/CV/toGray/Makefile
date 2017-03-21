CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main : toGray.cu
	nvcc $(CFLAGS) toGray.cu $(LIBS) -o main.out
