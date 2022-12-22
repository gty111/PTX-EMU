cfd: euler3d.cu
	@nvcc $(NVCC_FLARG) -I./inc $^ -o $(PTX_EMU_PATH)/bin/$@ 1>/dev/null 2>&1
	@$(PTX_EMU_PATH)/bin/$@ fvcorr.domn.097K 2>&1 1>/dev/null

.PHONY: cfd