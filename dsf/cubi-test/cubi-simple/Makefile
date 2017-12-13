TARGET = testCubi
CC     = g++
LFLAGS = -o $(TARGET) -lgmp -lm
NVCC = nvcc
CUDA_FLAGS = -gencode arch=compute_52,code=sm_52 

make:
	$(CC) $(TARGET).c $(LFLAGS)

clean: 
	rm -rf *o $(TARGET)
