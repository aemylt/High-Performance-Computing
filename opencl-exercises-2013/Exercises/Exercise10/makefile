#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#     make clean    ... remove object and executable files.
#

# verify that you are including the right make.def file for the platform
include ../../make.def

COMMON      = ../../Cpp_common

EXES =    pi$(EXE) 


all: $(EXES)

pi$(EXE): pi.$(OBJ)  
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o pi$(EXE) pi.$(OBJ) \
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

