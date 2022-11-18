/**
 * @author gtyinstinct
 * test lexer parser semantic
*/

#include"ptx-semantic.h"

// #define TOKEN
// #define TREE
#define SEMANTIC

int main(int argc, const char* argv[]) {
  assert(argc>=2);
  std::ifstream stream;
  stream.open(argv[1]);
  ANTLRInputStream input(stream);

  ptxLexer lexer(&input);
  CommonTokenStream tokens(&lexer);

  tokens.fill();

 #ifdef TOKEN
  // output tokens
  for (auto token : tokens.getTokens()) {
    std::cout << token->toString() << std::endl;
  }
 #endif

  ptxParser parser(&tokens);
  PtxListener tl;
  parser.addParseListener(&tl);

  tree::ParseTree *tree = parser.ast();
 
 #ifdef TREE
  // output grammar tree
  std::cout << tree->toStringTree(&parser) << std::endl << std::endl;
 #endif

 #ifdef SEMANTIC
  // output semantic
  tl.test_semantic();
 #endif

  return 0;
}