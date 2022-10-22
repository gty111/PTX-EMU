INCLUDE_DIR = $(shell pwd) $(shell spack location -i cuda@11.7)/include
NAME ?= test-gpusim
ARCH ?= sm_80
# -maxrregcount 16 -Xptxas=-v -src-in-ptx -lineinfo
NVCC_ARG = -arch=$(ARCH) -lcudart
NCU_OUT = ncu_out
NCU_ARG = -f -o $(NCU_OUT)/$(NAME) --set full --import-source on 
CUOBJDUMP_OUT = sass
CUOBJDUMP_ARG = -sass -ptx
CUOBJDUMP_OUTFILE = $(CUOBJDUMP_OUT)/$(NAME)
SUFFIX ?= cpp
LIB_OUT = libcudart.so.11.0

lib: src/$(NAME).$(SUFFIX)
	g++ $(addprefix -I,$(INCLUDE_DIR)) -fPIC -shared -Wl,--version-script=linux-so-version.txt $^ -o lib/$(LIB_OUT)

# export LD_LIBRARY_PATH=~/SIM_ON_GPU/lib:$LD_LIBRARY_PATH

bin: src/$(NAME).$(SUFFIX)
	$(shell [ ! -d bin ] && mkdir bin)
	nvcc $(NVCC_ARG) $(addprefix -I,$(INCLUDE_DIR)) $^ -o bin/$(NAME)
run: bin
	bin/$(NAME)
ncu: bin
	$(shell [ ! -d $(NCU_OUT) ] && mkdir $(NCU_OUT))
	ncu $(NCU_ARG) bin/$(NAME)
cuobjdump: bin/${NAME}
	$(shell [ ! -d $(CUOBJDUMP_OUT) ] && mkdir $(CUOBJDUMP_OUT)) 
	cuobjdump $(CUOBJDUMP_ARG) $^ >$(CUOBJDUMP_OUTFILE)

.PHONY: lib link bin run ncu cuobjdump
