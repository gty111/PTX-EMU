backprop: backprop.c facetrain.c imagenet.c backprop_cuda.cu
	@nvcc $(NVCC_FLARG) -DGPU $^ -o $(PTX_EMU_PATH)/bin/$@ 2>&1 1>/dev/null
	@$(PTX_EMU_PATH)/bin/$@ 65536 2>&1 1>/dev/null
	@diff result result_ref

.PHONY: backprop