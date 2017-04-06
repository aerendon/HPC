CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

main : mat_mult.cu
	nvcc $(CFLAGS) tiles.cu $(LIBS) -o main.out
