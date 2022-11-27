SRC = src
BENCHMARK = benchmark
INCLUDE_DIR = ${CUDA_PATH}/include \
	${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/runtime/src $(SRC)/build
LINK_DIR = ${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/run/usr/local/lib/
ARCH = sm_80
NVCC_FLARG = -arch=$(ARCH) -use_fast_math -lcudart
LIB_OUT = libcudart.so.11.0
CPP_FLAG = -std=c++2a -pthread -fPIC -shared -Wl,--version-script=linux-so-version.txt \
	$(SRC)/PTXEMU.cpp $(SRC)/build/*.cpp $(addprefix -I,$(INCLUDE_DIR)) $(addprefix -L,$(LINK_DIR)) \
	-lantlr4-runtime -o lib/$(LIB_OUT) 


CUSRC = $(wildcard $(BENCHMARK)/*.cu)

MINITEST = dummy dummy-add dummy-float dummy-grid dummy-mul dummy-sub dummy-condition \
		  dummy-long dummy-sieve dummy-share

TOTTEST = $(MINITEST) simpleGEMM-int simpleGEMM-float simpleGEMM-double \
		  simpleCONV-int simpleCONV-float simpleCONV-double 2Dentropy \
		  aligned-types all-pairs-distance bitonic

COLOR_RED   = \033[1;31m
COLOR_GREEN = \033[1;32m
COLOR_NONE  = \033[0m

all:$(CUSRC)

bin/%:$(BENCHMARK)/%.cu
	$(shell [ ! -d bin ] && mkdir bin)
	@nvcc $(NVCC_FLARG) $^ -o $@
	@cuobjdump -xptx $(patsubst $(BENCHMARK)/%.cu,%,$^).1.$(ARCH).ptx $@ > /dev/null
	@mv $(patsubst $(BENCHMARK)/%.cu,%,$^).1.$(ARCH).ptx \
		$(BENCHMARK)/$(patsubst $(BENCHMARK)/%.cu,%,$^).ptx

minitest:$(MINITEST)

test:$(TOTTEST) 

$(TOTTEST):%:bin/%
	@printf "[%20s]" $@ ;
	@if bin/$@ 2>&1 1>/dev/null ; then \
	printf " $(COLOR_GREEN)PASS$(COLOR_NONE)\n" ; \
	else \
	printf " $(COLOR_RED)FAIL$(COLOR_NONE)\n"; \
	fi

lib:
	$(shell [ ! -d lib ] && mkdir lib)
	make -C $(SRC)
	g++ -O3 $(CPP_FLAG) 
Dlib:
	$(shell [ ! -d lib ] && mkdir lib)
	make -C $(SRC)
	g++ -g -O0 $(CPP_FLAG)

Slib:
	$(shell [ ! -d lib ] && mkdir lib)
	make -C $(SRC)
	g++ -g -O0 -D DEBUGINTE -D LOGINTE $(CPP_FLAG)

# export LD_LIBRARY_PATH=~/SIM_ON_GPU/lib:$LD_LIBRARY_PATH
.PHONY: lib Dlib