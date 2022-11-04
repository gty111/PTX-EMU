#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <queue>

#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptxParserBaseListener.h"

using namespace ptxparser;
using namespace antlr4;

enum Qualifier{
  Q_NONE,
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
        std::string regMajorName;
        std::string regMinorName;
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
    void *statement;
    class REG {
      public:
        Qualifier regDataType;
        std::string regMajorName,regMinorName;
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

class PtxListener : public ptxParserBaseListener{
  public: 
    PtxContext ptxContext;
    KernelContext *kernelContext;
    ParamContext *paramContext;
    std::queue<Qualifier> qualifier;
    StatementType statementType;
    StatementContext statementContext;
    void *statement;
    std::queue<std::string> regMajorName,regMinorName;

    /* helper function */

    int extractIdx(std::string s){
      int ret = 0;
      for(char c:s){
        if(c>='0'&&c<='9'){
          ret = ret*10 + c - '0';
        }
      }
      return ret;
    }

    std::string extractName(std::string s){
      for(int i=0;i<s.size();i++){
        if(s[i]>='0'&&s[i]<='9'){
          return s.substr(0,i);
        }
      }
      return s;
    }

    /* listener function */

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
      /* performanceTuning */
      //TODO

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
      std::cout << "exit kernel " << "statement num:" << kernelContext->kernelStatements.size() << std::endl;
      delete kernelContext;
    }

    void enterQualifier(ptxParser::QualifierContext *ctx) override { 
      std::cout << "enter qualifier" << std::endl;
    }
    void exitQualifier(ptxParser::QualifierContext *ctx) override { 
      if(ctx->U64()){
        qualifier.push(Q_U64);
      }else if(ctx->U32()){
        qualifier.push(Q_U32);
      }else if(ctx->U16()){
        qualifier.push(Q_U16);
      }else if(ctx->U8()){
        qualifier.push(Q_U8);
      }else if(ctx->PRED()){
        qualifier.push(Q_PRED);
      }else if(ctx->B8()){
        qualifier.push(Q_B8);
      }else if(ctx->B16()){
        qualifier.push(Q_B16);
      }else if(ctx->B32()){
        qualifier.push(Q_B32);
      }else if(ctx->B64()){
        qualifier.push(Q_B64);
      }else if(ctx->F8()){
        qualifier.push(Q_F8);
      }else if(ctx->F16()){
        qualifier.push(Q_F16);
      }else if(ctx->F32()){
        qualifier.push(Q_F32);
      }else if(ctx->F64()){
        qualifier.push(Q_F64);
      }else if(ctx->S8()){
        qualifier.push(Q_S8);
      }else if(ctx->S16()){
        qualifier.push(Q_S16);
      }else if(ctx->S32()){
        qualifier.push(Q_S32);
      }else if(ctx->S64()){
        qualifier.push(Q_S64);
      }else if(ctx->V4()){
        qualifier.push(Q_V4);
      }else if(ctx->PARAM()){
        qualifier.push(Q_PARAM);
      }else if(ctx->GLOBAL()){
        qualifier.push(Q_GLOBAL);
      }else if(ctx->LOCAL()){
        qualifier.push(Q_LOCAL);
      }else if(ctx->SHARED()){
        qualifier.push(Q_SHARED);
      }else if(ctx->GT()){
        qualifier.push(Q_GT);
      }else if(ctx->GE()){
        qualifier.push(Q_GE);
      }else if(ctx->EQ()){
        qualifier.push(Q_EQ);
      }else if(ctx->NE()){
        qualifier.push(Q_NE);
      }else if(ctx->LT()){
        qualifier.push(Q_LT);
      }else if(ctx->TO()){
        qualifier.push(Q_TO);
      }else if(ctx->WIDE()){
        qualifier.push(Q_WIDE);
      }else if(ctx->SYNC()){
        qualifier.push(Q_SYNC);
      }else if(ctx->LO()){
        qualifier.push(Q_LO);
      }else if(ctx->UNI()){
        qualifier.push(Q_UNI);
      }else if(ctx->RN()){
        qualifier.push(Q_RN);
      }else if(ctx->A()){
        qualifier.push(Q_A);
      }else if(ctx->B()){
        qualifier.push(Q_B);
      }else if(ctx->D()){
        qualifier.push(Q_D);
      }else if(ctx->ROW()){
        qualifier.push(Q_ROW);
      }else if(ctx->ALIGNED()){
        qualifier.push(Q_ALIGNED);
      }else if(ctx->M8N8K4()){
        qualifier.push(Q_M8N8K4);
      }else if(ctx->M16N16K16()){
        qualifier.push(Q_M16N16K16);
      }
      std::cout << "exit qualifier " << qualifier.back() << std::endl;
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
      paramContext->paramType = qualifier.front();
      qualifier.pop();

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
      statementContext.statementType = statementType;
      statementContext.statement = statement;
      kernelContext->kernelStatements.push_back(statementContext);
      std::cout << "exit statement " << statementType << std::endl;
    }

    void enterRegStatement(ptxParser::RegStatementContext *ctx) override { 
      statement = new StatementContext::REG();
      std::cout << "enter regStatement" << std::endl;
    }
    void exitRegStatement(ptxParser::RegStatementContext *ctx) override { 
      auto st = (StatementContext::REG*) statement;
      /* qualifier */
      st->regDataType = qualifier.front();
      qualifier.pop();

      /* reg */
      st->regMajorName = regMajorName.front();
      regMajorName.pop();
      st->regMinorName = regMinorName.front();
      regMinorName.pop();

      /* digits */
      if(ctx->DIGITS()){
        st->regNum = stoi(ctx->DIGITS()->getText()) - 1;
      }else{
        st->regNum = 1;
      }

      /* end */
      statementType = S_REG;
      std::cout << "exit regStatement " << st->regDataType << ' '
          << st->regMajorName << st->regMinorName << ' ' 
          << st->regNum << std::endl;
    }

    void enterSharedStatement(ptxParser::SharedStatementContext *ctx) override { 
      statement = new StatementContext::SHARED();
      std::cout << "enter sharedStatement" << std::endl;
    }
    void exitSharedStatement(ptxParser::SharedStatementContext *ctx) override { 
      auto st = (StatementContext::SHARED *)statement;

      /* align */
      st->sharedAlign = stoi(ctx->DIGITS(0)->getText());

      /* qualifier */
      st->sharedDataType = qualifier.front();
      qualifier.pop();

      /* ID */
      st->sharedName = ctx->ID()->getText();

      /* size */
      st->sharedSize = stoi(ctx->DIGITS(1)->getText());

      /* end */
      statementType = S_SHARED;
      std::cout << "exit sharedStatement" << std::endl;
    }

    void enterLocalStatement(ptxParser::LocalStatementContext *ctx) override { 
      statement = new StatementContext::LOCAL();
      std::cout << "enter localStatement" << std::endl;
    }
    void exitLocalStatement(ptxParser::LocalStatementContext *ctx) override { 
      auto st = (StatementContext::LOCAL *)statement;

      /* align */
      st->localAlign = stoi(ctx->DIGITS(0)->getText());

      /* qualifier */
      st->localDataType = qualifier.front();
      qualifier.pop();

      /* ID */
      st->localName = ctx->ID()->getText();

      /* size */
      st->localSize = stoi(ctx->DIGITS(1)->getText());

      /* end */
      statementType = S_LOCAL;
      std::cout << "exit localStatement" << std::endl;
    }

    void enterDollorStatement(ptxParser::DollorStatementContext *ctx) override { 
      statement = new StatementContext::DOLLOR();
      std::cout << "enter dolorStatement" << std::endl;
    }
    void exitDollorStatement(ptxParser::DollorStatementContext *ctx) override { 
      auto st = (StatementContext::DOLLOR *)statement;

      /* ID */
      st->dollorName = ctx->ID()->getText();

      /* end */
      statementType = S_DOLLOR;
      std::cout << "exit dolorStatement" << std::endl;
    }

    void enterAtStatement(ptxParser::AtStatementContext *ctx) override { 
      statement = new StatementContext::AT();
      std::cout << "enter atStatement" << std::endl;
    }
    void exitAtStatement(ptxParser::AtStatementContext *ctx) override { 
      auto st = (StatementContext::AT *)statement;

      /* reg */
      st->op = *new OperandContext();
      st->op.opType = O_REG;
      st->op.operandContext = new OperandContext::REG();
      auto op = (OperandContext::REG*)st->op.operandContext;
      op->regIdx = extractIdx(regMajorName.front());
      op->regMajorName = extractName(regMajorName.front());
      regMajorName.pop();
      op->regMinorName = regMinorName.front();
      regMinorName.pop();

      /* ID */
      st->atLabelName = ctx->ID()->getText();

      /* end */
      statementType = S_AT;
      std::cout << "exit atStatement " << op->regMajorName << ' ' << op->regIdx << 
        ' ' << op->regMinorName << ' ' << st->atLabelName << std::endl;
    }

    void enterPragmaStatement(ptxParser::PragmaStatementContext *ctx) override { 
      statement = new StatementContext::PRAGMA();
      std::cout << "enter pragmaStatement" << std::endl;
    }
    void exitPragmaStatement(ptxParser::PragmaStatementContext *ctx) override { 
      auto st = (StatementContext::PRAGMA *)statement;

      /* prama string */
      st->pragmaString = ctx->STRING()->getText();

      /* end */
      statementType = S_PRAGMA;
      std::cout << "exit pragmaStatement" << std::endl;
    }

    void enterRetStatement(ptxParser::RetStatementContext *ctx) override { 
      statement = new StatementContext::RET();
      std::cout << "enter retStatement" << std::endl;
    }
    void exitRetStatement(ptxParser::RetStatementContext *ctx) override { 
      auto st = (StatementContext::RET *)statement;
      /* end */
      statementType = S_RET;
      std::cout << "exit retStatement" << std::endl;
    }

    void enterBarStatement(ptxParser::BarStatementContext *ctx) override { 
      statement = new StatementContext::BAR();
      std::cout << "enter barStatement" << std::endl;
    }
    void exitBarStatement(ptxParser::BarStatementContext *ctx) override {
      auto st = (StatementContext::BAR *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->barQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* DIGITS */
      st->barNum = stoi(ctx->DIGITS()->getText());

      /* end */
      statementType = S_BAR; 
      std::cout << "exit barStatement" << std::endl;
    }

    void enterBraStatement(ptxParser::BraStatementContext *ctx) override { 
      statement = new StatementContext::BRA();
      std::cout << "enter braStatement" << std::endl;
    }
    void exitBraStatement(ptxParser::BraStatementContext *ctx) override {
      auto st = (StatementContext::BRA *)statement; 

      /* qualifier */
      st->braQualifier = Q_NONE;
      if(qualifier.size()){
        st->braQualifier = qualifier.front();
        qualifier.pop();
      }

      /* ID */
      st->braLabel = ctx->ID()->getText();

      /* end */
      statementType = S_BRA;
      std::cout << "exit braStatement" << std::endl;
    }

    void enterRcpStatement(ptxParser::RcpStatementContext *ctx) override { 
      statement = new StatementContext::RCP();
      std::cout << "enter rcpStatement" << std::endl;
    }
    void exitRcpStatement(ptxParser::RcpStatementContext *ctx) override { 
      auto st = (StatementContext::RCP *)statement;
      /* end */
      statementType = S_RCP;
      std::cout << "exit rcpStatement" << std::endl;
    }

    void enterLdStatement(ptxParser::LdStatementContext *ctx) override { 
      statement = new StatementContext::LD();
      std::cout << "enter ldStatement" << std::endl;
    }
    void exitLdStatement(ptxParser::LdStatementContext *ctx) override {
      auto st = (StatementContext::LD *)statement;
      /* end */
      statementType = S_LD; 
      std::cout << "exit ldStatement" << std::endl;
    }

    void enterMovStatement(ptxParser::MovStatementContext *ctx) override { 
      statement = new StatementContext::MOV();
      std::cout << "enter movStatement" << std::endl;
    }
    void exitMovStatement(ptxParser::MovStatementContext *ctx) override { 
      auto st = (StatementContext::MOV *)statement;
      /* end */
      statementType = S_MOV;
      std::cout << "exit movStatement" << std::endl;
    }

    void enterSetpStatement(ptxParser::SetpStatementContext *ctx) override {
      statement = new StatementContext::SETP(); 
      std::cout << "enter setpStatement" << std::endl;
    }
    void exitSetpStatement(ptxParser::SetpStatementContext *ctx) override { 
      auto st = (StatementContext::SETP *)statement;
      /* end */
      statementType = S_SETP;
      std::cout << "exit setpStatement" << std::endl;
    }

    void enterCvtaStatement(ptxParser::CvtaStatementContext *ctx) override { 
      statement = new StatementContext::CVTA();
      std::cout << "enter cvtaStatement" << std::endl;
    }
    void exitCvtaStatement(ptxParser::CvtaStatementContext *ctx) override {
      auto st = (StatementContext::CVTA *)statement;
      /* end */
      statementType = S_CVTA;
      std::cout << "exit cvtaStatement" << std::endl;
    }

    void enterCvtStatement(ptxParser::CvtStatementContext *ctx) override {
      statement = new StatementContext::CVT(); 
      std::cout << "enter cvtStatement" << std::endl;
    }
    void exitCvtStatement(ptxParser::CvtStatementContext *ctx) override { 
      auto st = (StatementContext::CVT *)statement;
      /* end */
      statementType = S_CVT;
      std::cout << "exit cvtStatement" << std::endl;
    }

    void enterMulStatement(ptxParser::MulStatementContext *ctx) override { 
      statement = new StatementContext::MUL();
      std::cout << "enter mulStatement" << std::endl;
    }
    void exitMulStatement(ptxParser::MulStatementContext *ctx) override { 
      auto st = (StatementContext::MUL *)statement;
      /* end */
      statementType = S_MUL;
      std::cout << "exit mulStatement" << std::endl;
    }

    void enterDivStatement(ptxParser::DivStatementContext *ctx) override { 
      statement = new StatementContext::DIV();
      std::cout << "enter divStatement" << std::endl;
    }
    void exitDivStatement(ptxParser::DivStatementContext *ctx) override { 
      auto st = (StatementContext::DIV *)statement;
      /* end */
      statementType = S_DIV;
      std::cout << "exit divStatement" << std::endl;
    }

    void enterSubStatement(ptxParser::SubStatementContext *ctx) override { 
      statement = new StatementContext::SUB();
      std::cout << "enter subStatement" << std::endl;
    }
    void exitSubStatement(ptxParser::SubStatementContext *ctx) override { 
      auto st = (StatementContext::SUB *)statement;
      /* end */
      statementType = S_SUB;
      std::cout << "exit subStatement" << std::endl;
    }

    void enterAddStatement(ptxParser::AddStatementContext *ctx) override { 
      statement = new StatementContext::ADD();
      std::cout << "enter addStatement" << std::endl;
    }
    void exitAddStatement(ptxParser::AddStatementContext *ctx) override { 
      auto st = (StatementContext::ADD *)statement;
      /* end */
      statementType = S_ADD;
      std::cout << "exit addStatement" << std::endl;
    }

    void enterShlStatement(ptxParser::ShlStatementContext *ctx) override { 
      statement = new StatementContext::SHL();
      std::cout << "enter shlStatement" << std::endl;
    }
    void exitShlStatement(ptxParser::ShlStatementContext *ctx) override { 
      auto st = (StatementContext::SHL *)statement;
      /* end */
      statementType = S_SHL;
      std::cout << "exit shlStatement" << std::endl;
    }

    void enterShrStatement(ptxParser::ShrStatementContext *ctx) override { 
      statement = new StatementContext::SHR();
      std::cout << "enter shrStatement" << std::endl;
    }
    void exitShrStatement(ptxParser::ShrStatementContext *ctx) override { 
      auto st = (StatementContext::SHR *)statement;
      /* end */
      statementType = S_SHR;
      std::cout << "exit shrStatement" << std::endl;
    }

    void enterMaxStatement(ptxParser::MaxStatementContext *ctx) override { 
      statement = new StatementContext::MAX();
      std::cout << "enter maxStatement" << std::endl;
    }
    void exitMaxStatement(ptxParser::MaxStatementContext *ctx) override { 
      auto st = (StatementContext::MAX *)statement;
      /* end */
      statementType = S_MAX;
      std::cout << "exit maxStatement" << std::endl;
    }

    void enterMinStatement(ptxParser::MinStatementContext *ctx) override { 
      statement = new StatementContext::MIN();
      std::cout << "enter minStatement" << std::endl;
    }
    void exitMinStatement(ptxParser::MinStatementContext *ctx) override { 
      auto st = (StatementContext::MIN *)statement;
      /* end */
      statementType = S_MIN;
      std::cout << "exit minStatement" << std::endl;
    }

    void enterAndStatement(ptxParser::AndStatementContext *ctx) override { 
      statement = new StatementContext::AND();
      std::cout << "enter andStatement" << std::endl;
    }
    void exitAndStatement(ptxParser::AndStatementContext *ctx) override { 
      auto st = (StatementContext::AND *)statement;
      /* end */
      statementType = S_AND;
      std::cout << "exit andStatement" << std::endl;
    }

    void enterOrStatement(ptxParser::OrStatementContext *ctx) override { 
      statement = new StatementContext::OR();
      std::cout << "enter orStatement" << std::endl;
    }
    void exitOrStatement(ptxParser::OrStatementContext *ctx) override { 
      auto st = (StatementContext::OR *)statement;
      /* end */
      statementType = S_OR;
      std::cout << "exit orStatement" << std::endl;
    }

    void enterStStatement(ptxParser::StStatementContext *ctx) override { 
      statement = new StatementContext::ST();
      std::cout << "enter stStatement" << std::endl;
    }
    void exitStStatement(ptxParser::StStatementContext *ctx) override { 
      auto st = (StatementContext::ST *)statement;
      /* end */
      statementType = S_ST;
      std::cout << "exit stStatement" << std::endl;
    }

    void enterSelpStatement(ptxParser::SelpStatementContext *ctx) override { 
      statement = new StatementContext::SELP();
      std::cout << "enter selpStatement" << std::endl;
    }
    void exitSelpStatement(ptxParser::SelpStatementContext *ctx) override { 
      auto st = (StatementContext::SELP *)statement;
      /* end */
      statementType = S_SELP;
      std::cout << "exit selpStatement" << std::endl;
    }

    void enterMadStatement(ptxParser::MadStatementContext *ctx) override { 
      statement = new StatementContext::MAD();
      std::cout << "enter madStatement" << std::endl;
    }
    void exitMadStatement(ptxParser::MadStatementContext *ctx) override { 
      auto st = (StatementContext::MAD *)statement;
      /* end */
      statementType = S_MAD;
      std::cout << "exit madStatement" << std::endl;
    }

    void enterFmaStatement(ptxParser::FmaStatementContext *ctx) override { 
      statement = new StatementContext::FMA();
      std::cout << "enter fmaStatement" << std::endl;
    }
    void exitFmaStatement(ptxParser::FmaStatementContext *ctx) override {
      auto st = (StatementContext::FMA *)statement; 
      /* end */
      statementType = S_FMA;
      std::cout << "exit fmaStatement" << std::endl;
    }

    void enterWmmaStatement(ptxParser::WmmaStatementContext *ctx) override { 
      statement = new StatementContext::WMMA();
      std::cout << "enter wmmaStatement" << std::endl;
    }
    void exitWmmaStatement(ptxParser::WmmaStatementContext *ctx) override {
      auto st = (StatementContext::WMMA *)statement;
      /* end */
      statementType = S_WMMA; 
      std::cout << "exit wmmaStatement" << std::endl;
    }

    void enterReg(ptxParser::RegContext *ctx) override { 
      std::cout << "enter reg" << std::endl;
    }
    void exitReg(ptxParser::RegContext *ctx) override { 
      regMajorName.push(ctx->ID(0)->getText());
      if(ctx->ID().size()==2){
        regMinorName.push(ctx->ID(1)->getText());
      }else {
        regMinorName.push("");
      }
      std::cout << "exit reg " << regMajorName.back() << regMinorName.back() << std::endl;
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
  PtxListener tl;
  parser.addParseListener(&tl);

  tree::ParseTree *tree = parser.ast();
  
  std::cout << tree->toStringTree(&parser) << std::endl << std::endl;

  return 0;
}