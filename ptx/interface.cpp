#include <iostream>

#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"

using namespace ptxparser;
using namespace antlr4;

int main(int argc, const char* argv[]) {
  std::ifstream stream;
  stream.open(argv[1]);
  ANTLRInputStream input(stream);

  ptxLexer lexer(&input);
  CommonTokenStream tokens(&lexer);

  tokens.fill();
  for (auto token : tokens.getTokens()) {
    std::cout << token->toString() << std::endl;
  }

  ptxParser parser(&tokens);
  tree::ParseTree *tree = parser.ast();
  
  std::cout << tree->toStringTree(&parser) << std::endl << std::endl;

  return 0;
}