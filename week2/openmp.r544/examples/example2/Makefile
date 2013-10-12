#
#

EXE1=serial_pi.exe
EXE2=parallel_shared_pi.exe
EXE3=parallel_array_pi.exe
EXE4=parallel_private_pi.exe
EXE5=reduction_pi.exe

EXES=$(EXE1) $(EXE2) $(EXE3) $(EXE4) $(EXE5)

# compilers and flags
# by default we will choose GNU
COMP=GNU

ifeq ($(COMP), GNU)
  FC=gfortran
  CC=gcc
  FFLAGS=-fopenmp
  CFLAGS=-fopenmp
endif

# settings for PGI compilers
ifeq ($(COMP), PGI)
  FC=pgf90
  CC=pgcc
  FFLAGS=-mp
  CFLAGS=-mp
endif

# settings for Intel compilers
ifeq ($(COMP), INTEL)
  FC=ifort
  CC=icc
  FFLAGS=-openmp
  CFLAGS=-openmp
endif

all: $(EXES)

$(EXES): %.exe : %.c
	$(CC) $(CFLAGS) -lm $^ -o $@

.PHONY: all clean spotless

clean:
	\rm -f $(EXES)
