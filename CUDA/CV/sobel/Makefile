CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: seq seqOCV par parOCV

par: par.cu
	nvcc $(CFLAGS) par.cu $(LIBS) -o par.out

parOCV: parOCV.cu
	nvcc $(CFLAGS) parOCV.cu $(LIBS) -o parOCV.out

seqOCV: seqOCV
	g++ $(CFLAGS) seqOCV.cpp $(LIBS) -o seqOCV.out

seq:
	g++ $(CFLAGS) seq.cpp $(LIBS) -o seq.out
