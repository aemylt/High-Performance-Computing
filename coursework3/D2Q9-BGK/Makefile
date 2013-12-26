# Makefile

EXE1=d2q9-bgk.exe
EXES=$(EXE1)

CC=g++
TAU=tau_cc.sh
CFLAGS=-lm -Wall -O3 -fopenmp

all: $(EXES)

$(EXES): %.exe : %.cpp
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: all clean

clean:
	\rm -f $(EXES)

