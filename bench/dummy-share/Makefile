dummy-share: dummy-share.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@$(PTX_EMU_PATH)/bin/$@

.PHONY:dummy-share