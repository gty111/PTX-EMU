if [ ! -n "$PTX_EMU_SETUP" ]
then
    echo "please setup PTX_EMU environment first!"
    return
fi

if [ -d ${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/run/usr/local/lib ]
then
    export ANTLR_CPP_RUNTIME=1
    return
fi

cd ${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source
rm -rf build run
mkdir build && mkdir run && cd build
cmake ..
if [ $? -ne 0 ] ; then
    return
fi 
make -j 8
if [ $? -ne 0 ] ; then
    return
fi 
DESTDIR=${PTX_EMU_PATH}/antlr4/antlr4-cpp-runtime-4.11.1-source/run make install
export ANTLR_CPP_RUNTIME=1