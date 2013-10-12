# Makefile

EXE1=d2q9-bgk.exe
EXES=$(EXE1)

CC=tau_cc.sh
CFLAGS=-lm -Wall -DDEBUG

all: $(EXES)

$(EXES): %.exe : %.c
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: all clean

clean:
	\rm -f $(EXES)

