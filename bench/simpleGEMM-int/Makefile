simpleGEMM-int: simpleGEMM-int.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@$(PTX_EMU_PATH)/bin/$@

.PHONY:simpleGEMM-int