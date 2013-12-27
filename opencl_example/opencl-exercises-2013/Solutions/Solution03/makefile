#
#  USAGE:
#     make          ... to build the programs
#     make test     ... to run the default test case
#     make clean    ... to remove object and executable files
#

# verify that you are including the right make.def file for the platform
include ../../make.def

COMMON      = ../../Cpp_common

EXES     = vadd_abc$(EXE) 


all: $(EXES)

vadd_abc$(EXE): vadd_abc.$(OBJ) $(COMM_OBJS) 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o vadd_abc$(EXE) \
                         vadd_abc.$(OBJ) $(LIBS)

test: $(EXES)
	$(PRE)vadd_abc$(EXE)

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

