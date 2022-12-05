gemm_wmma: gemm_wmma.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@$(PTX_EMU_PATH)/bin/$@