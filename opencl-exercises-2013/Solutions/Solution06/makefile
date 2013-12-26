#
#  USAGE:
#     make          ... to build the programs
#     make test     ... to run the default test case
#     make clean    ... to remove object and executable files
#

# verify that you are including the right make.def file for the platform
include ../../make.def

COMMON      = ../../Cpp_common

EXES     = matmul_par_Crow$(EXE)

MMULOBJS =  matrix_lib.$(OBJ)

all: $(EXES)

matmul_par_Crow$(EXE): matmul_par_Crow.$(OBJ) $(MMULOBJS) 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o matmul_par_Crow$(EXE) \
                         matmul_par_Crow.$(OBJ) $(MMULOBJS)  $(LIBS)

test: $(EXES)
	$(PRE)matmul_par_Crow$(EXE)


clean:
	$(RM) $(EXES) *.$(OBJ)

veryclean:
	$(RM) $(EXES) *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<


