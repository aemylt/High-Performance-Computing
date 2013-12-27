#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#     make clean    ... remove object and executable files.
#

# verify that you are including the right make.def file for the platform
include make.def

COMMON      = Cpp_common

EXES =    d2q9-bgk$(EXE) 


all: $(EXES)

d2q9-bgk$(EXE): d2q9-bgk.$(OBJ)  
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o d2q9-bgk$(EXE) d2q9-bgk.$(OBJ) \
                         $(LIBS)

test: $(EXES)
	$(PRE)pi$(EXE);

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

