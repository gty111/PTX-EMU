BUILD_DIR="build"
BIN_DIR="bin"
TARGET="cpp"
LINKPATH="/root/antlr4/runtime/Cpp/run/usr/local/lib"
# grun ptx ast -gui
# export LD_LIBRARY_PATH=/root/antlr4/runtime/Cpp/run/usr/local/lib:$LD_LIBRARY_PATH
INCLUDEPATH="/root/antlr4/runtime/Cpp/runtime/src"
# -fPIC -shared
CPPARG="${BUILD_DIR}/*.cpp -g -std=c++2a -I${INCLUDEPATH} -L${LINKPATH} -lantlr4-runtime -pthread -o ${BIN_DIR}/ptx-parser"

rm -rf ${BUILD_DIR}/*
[ ! -d ${BUILD_DIR} ] && mkdir ${BUILD_DIR}

if [ ${TARGET} == "java" ] ; then
    echo "build target for" ${TARGET}
    antlr4 *.g4 -o ${BUILD_DIR}
    javac ${BUILD_DIR}/ptx*.java
elif [ ${TARGET} == "cpp" ] ; then
    echo "build target for" ${TARGET}
    [ ! -d ${BIN_DIR} ] && mkdir ${BIN_DIR}
    antlr4 -Dlanguage=Cpp *.g4 -listener -visitor -package ptxparser -o ${BUILD_DIR}
    cp test-ptx.cpp ptx-semantic.h ${BUILD_DIR}
    g++ ${CPPARG}
fi