CUDA_PATH=/usr/local/cuda
INCLUDE_DIR = $(shell pwd) ${CUDA_PATH}/include /root/antlr4/runtime/Cpp/runtime/src ptx/build
LINKPATH=/root/antlr4/runtime/Cpp/run/usr/local/lib
APP ?= test-gpusim
ARCH ?= sm_80
NVCC_FLARG = -arch=$(ARCH) -use_fast_math -lcudart
CPP_FLAG = -g -std=c++2a -pthread -fPIC -shared -Wl,--version-script=linux-so-version.txt
LIB_OUT = libcudart.so.11.0

CUSRC = $(wildcard src/*.cu)
#TARGETCU = $(patsubst src/%.cu,%,$(CUSRC))

TESTBIN = dummy dummy-add dummy-float dummy-grid dummy-mul dummy-sub dummy-condition \
		  dummy-long dummy-sieve dummy-share simpleGEMM-int simpleGEMM-float \
		  simpleGEMM-double

COLOR_RED   = \033[1;31m
COLOR_GREEN = \033[1;32m
COLOR_NONE  = \033[0m

all:$(CUSRC)

bin/%:src/%.cu
	$(shell [ ! -d bin ] && mkdir bin)
	nvcc $(NVCC_FLARG) $^ -o $@
	cuobjdump -xptx $(patsubst src/%.cu,%,$^).1.$(ARCH).ptx $@
	mv $(patsubst src/%.cu,%,$^).1.$(ARCH).ptx src/$(patsubst src/%.cu,%,$^).ptx

test:$(TESTBIN) 

$(TESTBIN):%:bin/%
	@printf "[%20s]" $@ ;
	@if bin/$@ 2>&1 1>/dev/null ; then \
	printf " $(COLOR_GREEN)PASS$(COLOR_NONE)\n" ; \
	else \
	printf " $(COLOR_RED)FAIL$(COLOR_NONE)\n"; \
	fi

lib: # ptx/PTXEMU.cpp ptx/build/*.cpp 
	$(shell [ ! -d lib ] && mkdir lib)
	make -C ptx
	g++ $(CPP_FLAG) ptx/PTXEMU.cpp ptx/build/*.cpp  $(addprefix -I,$(INCLUDE_DIR)) $(addprefix -L,$(LINKPATH)) -lantlr4-runtime -o lib/$(LIB_OUT)
Dlib:
	$(shell [ ! -d lib ] && mkdir lib)
	make -C ptx
	g++ -D DEBUGINTE -D LOGINTE $(CPP_FLAG) ptx/PTXEMU.cpp ptx/build/*.cpp $(addprefix -I,$(INCLUDE_DIR)) $(addprefix -L,$(LINKPATH)) -lantlr4-runtime -o lib/$(LIB_OUT)

# export LD_LIBRARY_PATH=~/SIM_ON_GPU/lib:$LD_LIBRARY_PATH
.PHONY: lib Dlib