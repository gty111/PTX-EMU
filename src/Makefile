antlr:
	$(shell rm -rf build && mkdir build)
	$(shell java -Xss500M -Xmx500M -cp "$$CLASSPATH" org.antlr.v4.Tool -Dlanguage=Cpp *.g4 -listener -visitor -package ptxparser -o build)

.PHONY: antlr