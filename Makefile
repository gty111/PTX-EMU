CUDA_PATH=/usr/local/cuda
INCLUDE_DIR = $(shell pwd) ${CUDA_PATH}/include /root/antlr4/runtime/Cpp/runtime/src ptx/build
LINKPATH=/root/antlr4/runtime/Cpp/run/usr/local/lib
APP ?= test-gpusim
ARCH ?= sm_80
# -maxrregcount 16 -Xptxas=-v -src-in-ptx -lineinfo
NVCC_FLARG = -arch=$(ARCH) -use_fast_math -lcudart
CPP_FLAG = -g -std=c++2a -pthread -fPIC -shared -Wl,--version-script=linux-so-version.txt
LIB_OUT = libcudart.so.11.0

CUSRC = $(wildcard src/*.cu)
TARGETCU = $(patsubst src/%.cu,%,$(CUSRC))


all:$(TARGETCU)

$(TARGETCU):%:src/%.cu
	$(shell [ ! -d bin ] && mkdir bin)
	nvcc $(NVCC_FLARG) $^ -o bin/$@
	cuobjdump -xptx $@.1.$(ARCH).ptx bin/$@
	mv $@.1.$(ARCH).ptx src/$@.ptx

lib: 
	$(shell [ ! -d lib ] && mkdir lib)
	make -C ptx
	g++ $(CPP_FLAG) ptx/PTXEMU.cpp ptx/build/*.cpp $(addprefix -I,$(INCLUDE_DIR)) $(addprefix -L,$(LINKPATH)) -lantlr4-runtime -o lib/$(LIB_OUT)
# export LD_LIBRARY_PATH=~/SIM_ON_GPU/lib:$LD_LIBRARY_PATH
.PHONY: lib link bin run ncu cuobjdump test setenv
