#
#  USAGE:
#     make          ... to build the programs
#     make test     ... to run the default test case
#     make clean    ... to remove object and executable files
#

# verify that you are including the right make.def file for the platform
include ../../make.def

COMMON      = ../../Cpp_common

EXES     = pi_ocl$(EXE) 


all: $(EXES)


pi_ocl$(EXE): pi_ocl.$(OBJ) 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o pi_ocl$(EXE) \
                         pi_ocl.$(OBJ) $(LIBS)


test: $(EXES)
	$(PRE)pi_ocl$(EXE)

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


