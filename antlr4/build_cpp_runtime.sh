if [ ! -n "$PTX_EMU_SETUP" ]
then
    echo "please setup PTX_EMU environment first!"
    return
fi

if [ ! -d ${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/run/usr/local/lib ]
then
    cd ${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source
    rm -rf build run
    mkdir build && mkdir run && cd build
    cmake .. -DANTLR_JAR_LOCATION=${PTX_EMU_PATH}/antlr4/antlr-4.11.1-complete.jar
    make
    DESTDIR=${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/run make install
fi