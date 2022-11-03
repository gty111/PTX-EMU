#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptxParserBaseListener.h"

using namespace ptxparser;
using namespace antlr4;

enum Qualifier{
  Q_U64,
  Q_U32,
  Q_U16,
  Q_U8,
  Q_PRED,
  Q_B8,
  Q_B16,
  Q_B32,
  Q_B64,
  Q_F8,
  Q_F16,
  Q_F32,
  Q_F64,
  Q_S8,
  Q_S16,
  Q_S32,
  Q_S64,
  Q_V4,
  Q_PARAM,
  Q_GLOBAL,
  Q_LOCAL,
  Q_SHARED,
  Q_GT,
  Q_GE,
  Q_EQ,
  Q_NE,
  Q_LT,
  Q_TO,
  Q_WIDE,
  Q_SYNC,
  Q_LO,
  Q_UNI,
  Q_RN,
  Q_A,
  Q_B,
  Q_D,
  Q_ROW,
  Q_ALIGNED,
  Q_M8N8K4,
  Q_M16N16K16
};

enum StatementType{
  S_REG,
  S_SHARED,
  S_LOCAL,
  S_DOLLOR,
  S_AT,
  S_PRAGMA,
  S_RET,
  S_BAR,
  S_BRA,
  S_RCP,
  S_LD,
  S_MOV,
  S_SETP,
  S_CVTA,
  S_CVT,
  S_MUL,
  S_DIV,
  S_SUB,
  S_ADD,
  S_SHL,
  S_SHR,
  S_MAX,
  S_MIN,
  S_AND,
  S_OR,
  S_ST,
  S_SELP,
  S_MAD,
  S_FMA,
  S_WMMA
};

enum OpType{
  O_REG,
  O_VAR,
  O_IMM,
  O_VEC
};

class OperandContext {
  public:
    OpType opType;
    void *operandContext;
    class REG {
      public:
        std::string regName;
        int regIdx;
    };
    class VAR {
      public:
        std::string varName;
    };
    class IMM {
      public:
        std::string immVal;
    };
    class VEC {
      public:
        std::vector<OperandContext> vec;
    };
};

class FetchAddressContext{
  public:
    std::string base;
    uint64_t offset;
};

class StatementContext{
  public:
    StatementType statementType;
    void *statementContext;
    class REG {
      public:
        Qualifier regDataType;
        std::string regName;
        int regNum;
    };
    class SHARED {
      public:
        int sharedAlign;
        Qualifier sharedDataType;
        std::string sharedName;
        int sharedSize;
    };
    class LOCAL {
      public:
        int localAlign;
        Qualifier localDataType;
        std::string localName;
        int localSize;
    };
    class DOLLOR {
      public:
        std::string dollorName;
    };
    class AT {
      public:
        OperandContext op;
        std::string atLabelName;
    };
    class PRAGMA {
      public:
        std::string pragmaString;
    };
    class RET {
    };
    class BAR {
      public:
        std::vector<Qualifier> barQualifier;
        int barNum;
    };
    class BRA {
      public:
        Qualifier braQualifier;
        std::string braLabel;
    };
    class RCP {
      public:
        std::vector<Qualifier> rcpQualifier;
        OperandContext rcpOp[2];
    };
    class LD {
      public:
        std::vector<Qualifier> ldQualifier;
        OperandContext ldOp;
        FetchAddressContext *fetchAddress;
    };
    class MOV {
      public:
        Qualifier movQualifier;
        OperandContext movOp[2];
    };
    class SETP {
      public:
        std::vector<Qualifier> setpQualifier;
        OperandContext setpOp[3];
    };
    class CVTA{
      public:
        std::vector<Qualifier> cvtaQualifier;
        OperandContext cvtaOp[2];
    };
    class CVT{
      public:
        std::vector<Qualifier> cvtQualifier;
        OperandContext cvtOp[2];
    };
    class MUL{
      public:
        std::vector<Qualifier> mulQualifier;
        OperandContext mulOp[3];
    };
    class DIV{
      public:
        std::vector<Qualifier> divQualifier;
        OperandContext divOp[3];
    };
    class SUB{
      public:
        std::vector<Qualifier> subQualifier;
        OperandContext subOp[3];
    };
    class ADD{
      public:
        std::vector<Qualifier> addQualifier;
        OperandContext addOp[3];
    };
    class SHL{
      public:
        std::vector<Qualifier> shlQualifier;
        OperandContext shlOp[3];
    };
    class SHR{
      public:
        std::vector<Qualifier> shrQualifier;
        OperandContext shrOp[3];
    };
    class MAX{
      public:
        std::vector<Qualifier> maxQualifier;
        OperandContext maxOp[3];
    };
    class MIN{
      public:
        std::vector<Qualifier> minQualifier;
        OperandContext minOp[3];
    };
    class AND{
      public:
        std::vector<Qualifier> andQualifier;
        OperandContext andOp[3];
    };
    class OR{
      public:
        std::vector<Qualifier> orQualifier;
        OperandContext orOp[3];
    };
    class ST{
      public:
        std::vector<Qualifier> stQualifier;
        OperandContext stOp;
        FetchAddressContext *fetchAddress;
    };
    class SELP{
      public:
        Qualifier selpQualifier;
        OperandContext selpOp[4];
    };
    class MAD{
      public:
        std::vector<Qualifier> madQualifier;
        OperandContext madOp[4];
    };
    class FMA{
      public:
        std::vector<Qualifier> fmaQualifier;
        OperandContext fmaOp[4];
    };
    class WMMA{
      public:
        std::vector<Qualifier> wmmaQualifier;
        std::vector<OperandContext> wmmaOp; 
    };
};

class ParamContext{
  public:
    Qualifier paramType;
    std::string paramName;
};

class KernelContext{
  public:
    bool ifVisibleKernel;
    bool ifEntryKernel;
    std::string kernelName;

    std::vector<ParamContext> kernelParams;
    std::vector<StatementContext> kernelStatements;
};

class PtxContext{
  public:
    int ptxMajorVersion; //done
    int ptxMinorVersion; //done
    int ptxTarget; //done
    int ptxAddressSize; //done

    std::vector<KernelContext> ptxKernels;
};

class TestListener : public ptxParserBaseListener{
  public: 
    PtxContext ptxContext;
    KernelContext *kernelContext;
    ParamContext *paramContext;
    Qualifier qualifier;

    void enterAst(ptxParser::AstContext *ctx) override{
      std::cout << "enter ast" << std::endl;
    }

    void exitAst(ptxParser::AstContext *ctx) override {
      std::cout << "exit ast" << std::endl;
    }
    void enterVersionDes(ptxParser::VersionDesContext *ctx) override { 
      std::cout << "enter versiondes" << std::endl;
    }
    void exitVersionDes(ptxParser::VersionDesContext *ctx) override { 
      auto digits = ctx->DIGITS();
      ptxContext.ptxMajorVersion = stoi(digits[0]->getText());
      ptxContext.ptxMinorVersion = stoi(digits[1]->getText());
      std::cout << "exit versiondes " << ptxContext.ptxMajorVersion
                << '.' << ptxContext.ptxMinorVersion << std::endl;
    }

    void enterTargetDes(ptxParser::TargetDesContext *ctx) override { 
      std::cout << "enter targetdes" << std::endl;
    }
    void exitTargetDes(ptxParser::TargetDesContext *ctx) override { 
      /* assume target always be 'sm_xx' */
      auto id = ctx->ID();
      auto str = id->getText();
      assert(str.length()==5);
      ptxContext.ptxTarget = stoi(id->getText().substr(3,2));
      std::cout << "exit targetdes " << ptxContext.ptxTarget << std::endl;
    }

    void enterAddressDes(ptxParser::AddressDesContext *ctx) override { 
      std::cout << "enter addressdes" << std::endl;
    }
    void exitAddressDes(ptxParser::AddressDesContext *ctx) override { 
      ptxContext.ptxAddressSize = stoi(ctx->DIGITS()->getText());
      std::cout << "exit addressdes " << ptxContext.ptxAddressSize << std::endl;
    }

    void enterKernels(ptxParser::KernelsContext *ctx) override { 
      std::cout << "enter kernels" << std::endl;
    }
    void exitKernels(ptxParser::KernelsContext *ctx) override { 
      std::cout << "exit kernels" << std::endl;
    }

    void enterKernel(ptxParser::KernelContext *ctx) override { 
      kernelContext = new KernelContext();
      std::cout << "enter kernel" << std::endl;
    }
    void exitKernel(ptxParser::KernelContext *ctx) override { 
      
      /* ID */
      kernelContext->kernelName = ctx->ID()->getText();
      std::cout << kernelContext->kernelName << std::endl;

      /* entry */
      if(ctx->ENTRY()){
        kernelContext->ifEntryKernel = true;
        std::cout << "entry" << std::endl;
      }else{
        kernelContext->ifEntryKernel = false;
      }

      /* visible */
      if(ctx->VISIBLE()){
        kernelContext->ifVisibleKernel = true;
        std::cout << "visible" << std::endl;
      }else{
        kernelContext->ifVisibleKernel = false;
      }

      /* end of parsing kernel */
      ptxContext.ptxKernels.push_back(*kernelContext);
      delete kernelContext;
      std::cout << "exit kernel" << std::endl;
    }

    void enterQualifier(ptxParser::QualifierContext *ctx) override { 
      std::cout << "enter qualifier" << std::endl;
    }
    void exitQualifier(ptxParser::QualifierContext *ctx) override { 
      if(ctx->U64()){
        qualifier = Q_U64;
      }else if(ctx->U32()){
        qualifier = Q_U32;
      }else if(ctx->U16()){
        qualifier = Q_U16;
      }else if(ctx->U8()){
        qualifier = Q_U8;
      }else if(ctx->PRED()){
        qualifier = Q_PRED;
      }else if(ctx->B8()){
        qualifier = Q_B8;
      }else if(ctx->B16()){
        qualifier = Q_B16;
      }else if(ctx->B32()){
        qualifier = Q_B32;
      }else if(ctx->B64()){
        qualifier = Q_B64;
      }else if(ctx->F8()){
        qualifier = Q_F8;
      }else if(ctx->F16()){
        qualifier = Q_F16;
      }else if(ctx->F32()){
        qualifier = Q_F32;
      }else if(ctx->F64()){
        qualifier = Q_F64;
      }else if(ctx->S8()){
        qualifier = Q_S8;
      }else if(ctx->S16()){
        qualifier = Q_S16;
      }else if(ctx->S32()){
        qualifier = Q_S32;
      }else if(ctx->S64()){
        qualifier = Q_S64;
      }else if(ctx->V4()){
        qualifier = Q_V4;
      }else if(ctx->PARAM()){
        qualifier = Q_PARAM;
      }else if(ctx->GLOBAL()){
        qualifier = Q_GLOBAL;
      }else if(ctx->LOCAL()){
        qualifier = Q_LOCAL;
      }else if(ctx->SHARED()){
        qualifier = Q_SHARED;
      }else if(ctx->GT()){
        qualifier = Q_GT;
      }else if(ctx->GE()){
        qualifier = Q_GE;
      }else if(ctx->EQ()){
        qualifier = Q_EQ;
      }else if(ctx->NE()){
        qualifier = Q_NE;
      }else if(ctx->LT()){
        qualifier = Q_LT;
      }else if(ctx->TO()){
        qualifier = Q_TO;
      }else if(ctx->WIDE()){
        qualifier = Q_WIDE;
      }else if(ctx->SYNC()){
        qualifier = Q_SYNC;
      }else if(ctx->LO()){
        qualifier = Q_LO;
      }else if(ctx->UNI()){
        qualifier = Q_UNI;
      }else if(ctx->RN()){
        qualifier = Q_RN;
      }else if(ctx->A()){
        qualifier = Q_A;
      }else if(ctx->B()){
        qualifier = Q_B;
      }else if(ctx->D()){
        qualifier = Q_D;
      }else if(ctx->ROW()){
        qualifier = Q_ROW;
      }else if(ctx->ALIGNED()){
        qualifier = Q_ALIGNED;
      }else if(ctx->M8N8K4()){
        qualifier = Q_M8N8K4;
      }else if(ctx->M16N16K16()){
        qualifier = Q_M16N16K16;
      }
      std::cout << "exit qualifier " << qualifier << std::endl;
    }

    void enterParams(ptxParser::ParamsContext *ctx) override { 
      std::cout << "enter params" << std::endl;
    }
    void exitParams(ptxParser::ParamsContext *ctx) override { 
      std::cout << "exit params" << std::endl;
    }

    void enterParam(ptxParser::ParamContext *ctx) override { 
      paramContext = new ParamContext();
      std::cout << "enter param" << std::endl;
    }
    void exitParam(ptxParser::ParamContext *ctx) override { 

      /* ID */
      paramContext->paramName = ctx->ID()->getText();

      /* qualifier */
      paramContext->paramType = qualifier;

      /* end of parsing param */
      kernelContext->kernelParams.push_back(*paramContext);
      std::cout << "exit param " << paramContext->paramName 
                << " " << paramContext->paramType << std::endl;
      delete paramContext;
    }

    void enterCompoundStatement(ptxParser::CompoundStatementContext *ctx) override { 
      std::cout << "enter compoundstatement" << std::endl;
    }
    void exitCompoundStatement(ptxParser::CompoundStatementContext *ctx) override { 
      std::cout << "exit compoundstatement" << std::endl;
    }

    void enterStatements(ptxParser::StatementsContext *ctx) override { 
      std::cout << "enter statements" << std::endl;
    }
    void exitStatements(ptxParser::StatementsContext *ctx) override { 
      std::cout << "exit statements" << std::endl;
    }

    void enterStatement(ptxParser::StatementContext *ctx) override { 
      std::cout << "enter statement" << std::endl;
    }
    void exitStatement(ptxParser::StatementContext *ctx) override { 
      std::cout << "exit statement" << std::endl;
    }

    void enterReg(ptxParser::RegContext *ctx) override { 
      std::cout << "enter reg" << std::endl;
    }
    void exitReg(ptxParser::RegContext *ctx) override { 
      std::cout << "exit reg" << std::endl;
    }

    void enterVector(ptxParser::VectorContext *ctx) override { 
      std::cout << "enter vector" << std::endl;
    }
    void exitVector(ptxParser::VectorContext *ctx) override { 
      std::cout << "exit vector" << std::endl;
    }

    void enterFetchAddress(ptxParser::FetchAddressContext *ctx) override { 
      std::cout << "enter fectchaddress" << std::endl;
    }
    void exitFetchAddress(ptxParser::FetchAddressContext *ctx) override { 
      std::cout << "exit fetchaddress" << std::endl;
    }
};

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
  TestListener tl;
  parser.addParseListener(&tl);

  tree::ParseTree *tree = parser.ast();
  
  std::cout << tree->toStringTree(&parser) << std::endl << std::endl;

  return 0;
}