TARGET = dsf-seq
CC     = mpicc
LFLAGS = -o $(TARGET) -lgmp -lm

make:
	$(CC) $(TARGET).c $(LFLAGS)

clean: 
	rm -rf *o $(TARGET)
