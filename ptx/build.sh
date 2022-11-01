BUILD_DIR="build"

rm ${BUILD_DIR}/*
antlr4 *.g4 -o ${BUILD_DIR}
javac ${BUILD_DIR}/ptx*.java