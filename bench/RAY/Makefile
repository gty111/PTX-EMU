RAY: EasyBMP.cpp makebmp.cpp rayTracing.cu 
	@nvcc -g $(NVCC_FLARG) -I. -I../inc -L../library -lcutil $^ -o $(PTX_EMU_PATH)/bin/$@ 1>/dev/null 2>&1
	@-rm output.bmp
	@$(PTX_EMU_PATH)/bin/$@ 512 512 1>/dev/null 2>&1

.PHONY:RAY

# TODO result check