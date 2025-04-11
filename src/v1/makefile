CC = gcc
CFLAGS = -Wall -pg -O2
PROF_OUT = gmon.out
ANALYSIS = analysis.txt

EXE = nn.exe
SRC = nn.c

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

gprof: $(EXE)
	gprof $(EXE) $(PROF_OUT) > $(ANALYSIS)
	@echo "Profiling results saved to $(ANALYSIS)"

clean:
	rm -f $(EXE) $(PROF_OUT) $(ANALYSIS)
