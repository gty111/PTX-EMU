antlr: *.g4
	$(shell rm -rf build && mkdir build)
	$(shell java -Xss500M -Xmx500M -cp "/usr/local/lib/antlr-4.11.1-complete.jar:$CLASSPATH" org.antlr.v4.Tool -Dlanguage=Cpp *.g4 -listener -visitor -package ptxparser -o build)
