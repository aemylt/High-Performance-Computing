#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#     make clean    ... remove object and executable files.
#

# verify that you are including the right make.def file for the platform
include ../../make.def

COMMON      = ../../Cpp_common

EXES     = matmul$(EXE)

MMULOBJS = matmul.$(OBJ) matrix_lib.$(OBJ)

all: $(EXES)

matmul$(EXE): $(MMULOBJS) 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o matmul$(EXE) $(MMULOBJS) \
                         $(LIBS)

test: $(EXES)
	$(PRE)matmul$(EXE)

clean:
	$(RM) $(EXES) *.$(OBJ) 

veryclean:
	$(RM) $(EXES)  *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<

