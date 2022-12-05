gemm_bmma: gemm_bmma.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@$(PTX_EMU_PATH)/bin/$@