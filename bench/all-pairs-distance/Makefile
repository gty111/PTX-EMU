all-pairs-distance: all-pairs-distance.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@$(PTX_EMU_PATH)/bin/$@