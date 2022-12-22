RAY: EasyBMP.cpp makebmp.cpp rayTracing.cu 
	@nvcc -use_fast_math $(NVCC_FLARG) -I. -I./inc -L./library -lcutil $^ -o $(PTX_EMU_PATH)/bin/$@ 1>/dev/null 2>&1
	@$(PTX_EMU_PATH)/bin/$@ 256 256 1>/dev/null 2>&1
	@-rm output.bmp

.PHONY:RAY

# TODO result check(you can view output.bmp to check)