bfs: bfs.cu
	@nvcc $(NVCC_FLARG) $^ -o $(PTX_EMU_PATH)/bin/$@
	@rm result.txt 
	@$(PTX_EMU_PATH)/bin/$@ graph65536.txt 1>/dev/null 2>&1
	@diff result.txt result_ref.txt 1>/dev/null 2>&1