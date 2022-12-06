SRC = src
INCLUDE_DIR = ${CUDA_PATH}/include \
	${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/runtime/src $(SRC)/build
LINK_DIR = ${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/run/usr/local/lib/
ARCH = sm_80
NVCC_FLARG = "-arch=$(ARCH) -use_fast_math -lcudart"
LIB_OUT = libcudart.so.11.0
CPP_FLAG = -std=c++2a -pthread -fPIC -shared -Wl,--version-script=linux-so-version.txt \
	$(SRC)/PTXEMU.cpp $(SRC)/build/*.cpp $(addprefix -I,$(INCLUDE_DIR)) $(addprefix -L,$(LINK_DIR)) \
	-lantlr4-runtime -o lib/$(LIB_OUT) 


BENCH = $(wildcard bench/*)

MINITEST = dummy dummy-add dummy-float dummy-grid dummy-mul dummy-sub dummy-condition \
		  dummy-long dummy-sieve dummy-share

TOTTEST = $(MINITEST) simpleGEMM-int simpleGEMM-float simpleGEMM-double \
		  simpleCONV-int simpleCONV-float simpleCONV-double 2Dentropy \
		  aligned-types all-pairs-distance bitonic bfs backprop RAY

COLOR_RED   = \033[1;31m
COLOR_GREEN = \033[1;32m
COLOR_NONE  = \033[0m

minitest:$(MINITEST)

test:$(TOTTEST) 

$(TOTTEST):%:
	@printf "[%20s]" $@ ;
	@if make -C bench/$@ NVCC_FLARG=$(NVCC_FLARG) ARCH=$(ARCH) 1>/dev/null 2>&1 ; then \
	printf " $(COLOR_GREEN)PASS$(COLOR_NONE)\n" ; \
	else \
	printf " $(COLOR_RED)FAIL$(COLOR_NONE)\n"; \
	fi
	@if [ ! -e bench/$@/$@.1.$(ARCH).ptx ] ; then \
	cuobjdump -xptx $@.1.$(ARCH).ptx bin/$@ > /dev/null ;\
	mv $@.1.$(ARCH).ptx bench/$@ ;\
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
	g++ -g -O0 -D LOGEMU -D DEBUGINTE -D LOGINTE $(CPP_FLAG)

.PHONY: lib Dlib $(BENCH)