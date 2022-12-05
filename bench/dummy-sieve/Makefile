dummy-sieve: dummy-sieve.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@$(PTX_EMU_PATH)/bin/$@

.PHONY:dummy-sieve