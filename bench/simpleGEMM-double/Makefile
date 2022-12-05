simpleGEMM-double: simpleGEMM-double.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@$(PTX_EMU_PATH)/bin/$@

.PHONY:simpleGEMM-double