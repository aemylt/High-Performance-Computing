#
# Makefile
#

CC=gcc
CFLAGS=-Wall -g
#CFLAGS+=-DDEBUG
#CFLAGS+=-O2
#CFLAGS+=-O2 -finline-functions
#CFLAGS+=-O3
#CFLAGS+=-O3 -ffast-math
CFLAGS+=-DTWISTER

LDFLAGS=-g

OBJ1=dartboard_pi.o
OBJ2=mersenne.o
OBJS=$(OBJ1) $(OBJ2)

EXE1=dartboard_pi.exe
EXES=$(EXE1)

all: $(EXES)

$(EXE1): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJS): %.o : %.c mersenne.h
	$(CC) -c $(CFLAGS) -o $@ $<

.PHONY: clean all

clean:
	\rm -f $(EXES) 
	\rm -f *.o
