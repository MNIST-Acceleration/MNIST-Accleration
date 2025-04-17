CC = nvcc
CFLAGS = -O3 -use_fast_math -Xcompiler "-Wall"
EXE = nn.exe
SRC = nn_cuda.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC)

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
