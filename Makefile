CUDA_PATH=/usr/local/cuda
INCLUDE_DIR = $(shell pwd) ${CUDA_PATH}/include /root/antlr4/runtime/Cpp/runtime/src ptx/build
LINKPATH=/root/antlr4/runtime/Cpp/run/usr/local/lib
APP ?= test-gpusim
ARCH ?= sm_80
# -maxrregcount 16 -Xptxas=-v -src-in-ptx -lineinfo
NVCC_FLARG = -arch=$(ARCH) -use_fast_math -lcudart
CPP_FLAG = -std=c++2a -pthread -fPIC -shared -Wl,--version-script=linux-so-version.txt
NCU_OUT = ncu_out
NCU_ARG = -f -o $(NCU_OUT)/$(APP) --set full --import-source on 
CUOBJDUMP_OUT = sass
CUOBJDUMP_ARG = -sass -ptx
CUOBJDUMP_OUTFILE = $(CUOBJDUMP_OUT)/$(APP)
LIB_OUT = libcudart.so.11.0

CUSRC = $(wildcard src/*.cu)
TARGETCU = $(patsubst src/%.cu,%,$(CUSRC))


all:$(TARGETCU)

$(TARGETCU):%:src/%.cu
	$(shell [ ! -d bin ] && mkdir bin)
	nvcc $(NVCC_FLARG) $^ -o bin/$@

run: bin
	bin/$(APP)
ncu: bin
	$(shell [ ! -d $(NCU_OUT) ] && mkdir $(NCU_OUT))
	ncu $(NCU_ARG) bin/$(APP)
cuobjdump: bin/${APP}
	$(shell [ ! -d $(CUOBJDUMP_OUT) ] && mkdir $(CUOBJDUMP_OUT)) 
	cuobjdump $(CUOBJDUMP_ARG) $^ >$(CUOBJDUMP_OUTFILE)

lib: 
	$(shell [ ! -d lib ] && mkdir lib)
	make -C ptx
	g++ $(CPP_FLAG) ptx/test-gpusim.cpp ptx/build/*.cpp $(addprefix -I,$(INCLUDE_DIR)) $(addprefix -L,$(LINKPATH)) -lantlr4-runtime -o lib/$(LIB_OUT)
# export LD_LIBRARY_PATH=~/SIM_ON_GPU/lib:$LD_LIBRARY_PATH
.PHONY: lib link bin run ncu cuobjdump test setenv
