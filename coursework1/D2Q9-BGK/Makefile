# Makefile

EXE1=d2q9-bgk.exe
EXES=$(EXE1)

TAU=tau_cc.sh
CC=gcc
CFLAGS=-fopenmp -O3 -lm -Wall

all: $(EXES)

$(EXES): %.exe : %.c
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: all clean

clean:
	\rm -f $(EXES)
