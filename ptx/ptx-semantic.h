/**
 * @author gtyinstinct
 * extract ptx semantic from grammar tree built by antlr4
*/
#ifndef __PTX_SEMANTIC__
#define __PTX_SEMANTIC__

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
  Q_V2,
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
  Q_M16N16K16,
  Q_NEU,
  Q_NC,
  Q_FTZ,
  Q_APPROX,
  Q_LTU,
  Q_LE,
  Q_GTU,
  Q_LEU,
  Q_DOTADD,
  Q_GEU,
  Q_RZI,
  Q_DOTOR
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
  S_WMMA,
  S_NEG,
  S_NOT,
  S_SQRT,
  S_COS,
  S_LG2,
  S_EX2,
  S_ATOM,
  S_XOR,
  S_ABS,
  S_SIN
};

enum OpType{
  O_REG,
  O_VAR,
  O_IMM,
  O_VEC,
  O_FA
};

enum WmmaType {
  WMMA_LOAD,
  WMMA_STORE,
  WMMA_MMA
};

class OperandContext {
  public:
    OpType opType;
    void *operand;
    class REG {
      public:
        std::string regMajorName;
        std::string regMinorName;
        int regIdx;

        REG(){}

        REG(const REG &reg){
          this->regIdx = reg.regIdx;
          this->regMajorName = reg.regMajorName;
          this->regMinorName = reg.regMinorName;
        }
    };
    class VAR {
      public:
        std::string varName;

        VAR(){}

        VAR(std::string s){
          varName = s;
        }
    };
    class IMM {
      public:
        std::string immVal;
        
        IMM(){}

        IMM(std::string s){
          immVal = s;
        }
    };
    class VEC {
      public:
        std::vector<OperandContext> vec;
      
        VEC(){}

        VEC(const VEC &v){
          this->vec = v.vec;
        }
    };
    class FA{ //fetch address
      public:
        std::string ID;
        OperandContext *reg;
        std::string offset;
        bool ifMinus;

        FA(){}

        FA(const FA &fa){
          this->ID = fa.ID;
          this->reg = fa.reg;
          this->offset  = fa.offset;
          this->ifMinus = fa.ifMinus;
        }
    };
};

class StatementContext{
  public:
    StatementType statementType;
    void *statement;
    class REG {
      public:
        std::vector<Qualifier> regDataType;
        std::string regName;
        int regNum;
    };
    class SHARED {
      public:
        int sharedAlign;
        std::vector<Qualifier> sharedDataType;
        std::string sharedName;
        int sharedSize;
    };
    class LOCAL {
      public:
        int localAlign;
        std::vector<Qualifier> localDataType;
        std::string localName;
        int localSize;
    };
    class DOLLOR {
      public:
        std::string dollorName;
    };
    class AT {
      public:
        OperandContext atOp;
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
        std::vector<Qualifier> braQualifier;
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
        OperandContext ldOp[2];
    };
    class MOV {
      public:
        std::vector<Qualifier> movQualifier;
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
        OperandContext stOp[2];
    };
    class SELP{
      public:
        std::vector<Qualifier> selpQualifier;
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
        WmmaType wmmaType;
        std::vector<Qualifier> wmmaQualifier;
        OperandContext wmmaOp[4]; 
    };
    class NEG{
      public:
        std::vector<Qualifier> negQualifier;
        OperandContext negOp[2];
    };
    class NOT{
      public:
        std::vector<Qualifier> notQualifier;
        OperandContext notOp[2];
    };
    class SQRT{
      public:
        std::vector<Qualifier> sqrtQualifier;
        OperandContext sqrtOp[2];
    };
    class COS{
      public:
        std::vector<Qualifier> cosQualifier;
        OperandContext cosOp[2];
    };
    class LG2{
      public:
        std::vector<Qualifier> lg2Qualifier;
        OperandContext lg2Op[2];
    };
    class EX2{
      public:
        std::vector<Qualifier> ex2Qualifier;
        OperandContext ex2Op[2];
    };
    class ATOM{
      public:
        std::vector<Qualifier> atomQualifier;
        OperandContext atomOp[4];
        int operandNum;
    };
    class XOR{
      public:
        std::vector<Qualifier> xorQualifier;
        OperandContext xorOp[3];
    };
    class ABS{
      public:
        std::vector<Qualifier> absQualifier;
        OperandContext absOp[2];
    };
    class SIN{
      public:
        std::vector<Qualifier> sinQualifier;
        OperandContext sinOp[2];
    };
};


class ParamContext{
  public:
    Qualifier paramType;
    std::string paramName;
    int paramAlign;
    int paramNum;
};

class KernelContext{
  public:
    bool ifVisibleKernel;
    bool ifEntryKernel;
    std::string kernelName;
    class Maxntid{
      public:
        int x,y,z;
    };
    Maxntid maxntid;
    int minnctapersm;

    std::vector<ParamContext> kernelParams;
    std::vector<StatementContext> kernelStatements;
};

class PtxContext{
  public:
    int ptxMajorVersion; 
    int ptxMinorVersion; 
    int ptxTarget; 
    int ptxAddressSize; 

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
    std::queue<OperandContext*> op;

    /* helper function */

    void test_semantic(){
      PtxContext &ptx = ptxContext;
      std::printf(".version %d.%d\n",ptx.ptxMajorVersion,ptx.ptxMinorVersion);
      std::printf(".target sm_%d\n",ptx.ptxTarget);
      std::printf(".address_size %d\n",ptx.ptxAddressSize);
      std::printf("number of kernel %d\n",ptx.ptxKernels.size());
      for(int i=0;i<ptx.ptxKernels.size();i++){
        KernelContext &kernel = ptx.ptxKernels[i];
        if(kernel.ifEntryKernel){
          std::printf(".entry ");
        }
        if(kernel.ifVisibleKernel){
          std::printf(".visible ");
        }
        std::printf("%s\n",kernel.kernelName.c_str());
        std::printf("number of param %d\n",kernel.kernelParams.size());
        for(int i=0;i<kernel.kernelParams.size();i++){
          ParamContext &param = kernel.kernelParams[i];
          std::printf("%s: ",param.paramName.c_str());
          if(param.paramAlign!=0){
            std::printf("align %d ",param.paramAlign);
          }
          std::printf("%s ",Q2s(param.paramType).c_str());
          if(param.paramNum!=0){
            std::printf("arraySize %d ",param.paramNum);
          }
          std::printf("\n");
        }
        std::printf("number of statements %d\n",kernel.kernelStatements.size());
        for(int i=0;i<kernel.kernelStatements.size();i++){
          StatementContext stat = kernel.kernelStatements[i];
          std::printf("%s %p\n",S2s(stat.statementType).c_str(),stat.statement);
        }
      }
    }

    void extractREG(std::string s,int &idx,std::string &name){
      int ret = 0;
      for(char c:s){
        if(c>='0'&&c<='9'){
          ret = ret*10 + c - '0';
        }
      }
      idx = ret;
      for(int i=0;i<s.size();i++){
        if((s[i]>='0'&&s[i]<='9')){
          name = s.substr(0,i);
          return;
        }
      }
      name = s;
    }

    void fetchOperand(OperandContext &oc){
      assert(op.size());
      oc.operand = op.front()->operand;
      oc.opType = op.front()->opType;
      op.pop();
    }

    static std::string Q2s(Qualifier q){
      switch(q){
      case Q_U64:return ".u64";
      case Q_U32:return ".u32";
      case Q_U16:return ".u16";
      case Q_U8:return ".u8";
      case Q_PRED:return ".pred";
      case Q_B8:return ".b8";
      case Q_B16:return ".b16";
      case Q_B32:return ".b32";
      case Q_B64:return ".b64";
      case Q_F8:return ".f8";
      case Q_F16:return ".f16";
      case Q_F32:return ".f32";
      case Q_F64:return ".f64";
      case Q_S8:return ".s8";
      case Q_S16:return ".s16";
      case Q_S32:return ".s32";
      case Q_S64:return ".s64";
      case Q_V2:return ".v2";
      case Q_V4:return ".v4";
      case Q_PARAM:return ".param";
      case Q_GLOBAL:return ".global";
      case Q_LOCAL:return ".local";
      case Q_SHARED:return ".shared";
      case Q_GT:return ".gt";
      case Q_GE:return ".ge";
      case Q_EQ:return ".eq";
      case Q_NE:return ".ne";
      case Q_LT:return ".lt";
      case Q_TO:return ".to";
      case Q_WIDE:return ".wide";
      case Q_SYNC:return ".sync";
      case Q_LO:return ".lo";
      case Q_UNI:return ".uni";
      case Q_RN:return ".rn";
      case Q_A:return ".a";
      case Q_B:return ".b";
      case Q_D:return ".d";
      case Q_ROW:return ".row";
      case Q_ALIGNED:return ".aligned";
      case Q_M8N8K4:return ".m8n8k4";
      case Q_M16N16K16:return ".m16n16k16";
      case Q_NEU:return ".neu";
      case Q_NC:return ".nc";
      case Q_FTZ:return ".ftz";
      case Q_APPROX:return ".approx";
      case Q_LTU:return ".ltu";
      case Q_LE:return ".le";
      case Q_GTU:return ".gtu";
      case Q_LEU:return ".leu";
      case Q_DOTADD:return ".add";
      case Q_GEU:return ".geu";
      case Q_RZI:return ".rzi";
      case Q_DOTOR:return ".or";
      default:assert(0);
      }
    }

    static std::string S2s(StatementType s){
      switch (s)
      {
      case S_REG:return "reg";
      case S_SHARED:return "shared";
      case S_LOCAL:return "local";
      case S_DOLLOR:return "$";
      case S_AT:return "@";
      case S_PRAGMA:return "pragma";
      case S_RET:return "ret";
      case S_BAR:return "bar";
      case S_BRA:return "bra";
      case S_RCP:return "rcp";
      case S_LD:return "ld";
      case S_MOV:return "mov";
      case S_SETP:return "setp";
      case S_CVTA:return "cvta";
      case S_CVT:return "cvt";
      case S_MUL:return "mul";
      case S_DIV:return "div";
      case S_SUB:return "sub";
      case S_ADD:return "add";
      case S_SHL:return "shl";
      case S_SHR:return "shr";
      case S_MAX:return "max";
      case S_MIN:return "min";
      case S_AND:return "and";
      case S_OR:return "or";
      case S_ST:return "st";
      case S_SELP:return "selp";
      case S_MAD:return "mad";
      case S_FMA:return "fma";
      case S_WMMA:return "wmma";
      case S_NEG:return "neg";
      case S_NOT:return "not";
      case S_SQRT:return "sqrt";
      case S_COS:return "cos";
      case S_LG2:return "lg2";
      case S_EX2:return "ex2";
      case S_ATOM:return "atom";
      case S_XOR:return "xor";
      case S_ABS:return "abs";
      case S_SIN:return "sin";
      default:assert(0);
      }
    }

    /* listener function */

    void enterAst(ptxParser::AstContext *ctx) override{
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void exitAst(ptxParser::AstContext *ctx) override {
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void enterVersionDes(ptxParser::VersionDesContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitVersionDes(ptxParser::VersionDesContext *ctx) override { 
      auto digits = ctx->DIGITS();
      ptxContext.ptxMajorVersion = stoi(digits[0]->getText());
      ptxContext.ptxMinorVersion = stoi(digits[1]->getText());
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterTargetDes(ptxParser::TargetDesContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitTargetDes(ptxParser::TargetDesContext *ctx) override { 
      /* assume target always be 'sm_xx' */
      auto id = ctx->ID();
      auto str = id->getText();
      assert(str.length()==5);
      ptxContext.ptxTarget = stoi(id->getText().substr(3,2));
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterAddressDes(ptxParser::AddressDesContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitAddressDes(ptxParser::AddressDesContext *ctx) override { 
      ptxContext.ptxAddressSize = stoi(ctx->DIGITS()->getText());
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterPerformanceTuning(ptxParser::PerformanceTuningContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitPerformanceTuning(ptxParser::PerformanceTuningContext *ctx) override { 
      /* init val */
      kernelContext->maxntid.x = 0;
      kernelContext->maxntid.y = 0;
      kernelContext->maxntid.z = 0;
      
      kernelContext->minnctapersm = 0;

      /* extrac val */
      if(ctx->MAXNTID()){
        kernelContext->maxntid.x = stoi(ctx->DIGITS(0)->getText());
        kernelContext->maxntid.y = stoi(ctx->DIGITS(1)->getText());
        kernelContext->maxntid.z = stoi(ctx->DIGITS(2)->getText());
      }else if(ctx->MINNCTAPERSM()){
        kernelContext->minnctapersm = stoi(ctx->DIGITS(0)->getText());
      }else assert(0 && "performancetuning not recognized!\n");

      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterKernels(ptxParser::KernelsContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitKernels(ptxParser::KernelsContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterKernel(ptxParser::KernelContext *ctx) override { 
      kernelContext = new KernelContext();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitKernel(ptxParser::KernelContext *ctx) override { 
      /* ID */
      kernelContext->kernelName = ctx->ID()->getText();

      /* entry */
      if(ctx->ENTRY()){
        kernelContext->ifEntryKernel = true;
      }else{
        kernelContext->ifEntryKernel = false;
      }

      /* visible */
      if(ctx->VISIBLE()){
        kernelContext->ifVisibleKernel = true;
      }else{
        kernelContext->ifVisibleKernel = false;
      }

      /* end of parsing kernel */
      ptxContext.ptxKernels.push_back(*kernelContext);
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
      kernelContext = nullptr;
    }


    void enterQualifier(ptxParser::QualifierContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
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
      }else if(ctx->V2()){
        qualifier.push(Q_V2);
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
      }else if(ctx->NEU()){
        qualifier.push(Q_NEU);
      }else if(ctx->NC()){
        qualifier.push(Q_NC);
      }else if(ctx->FTZ()){
        qualifier.push(Q_FTZ);
      }else if(ctx->APPROX()){
        qualifier.push(Q_APPROX);
      }else if(ctx->LTU()){
        qualifier.push(Q_LTU);
      }else if(ctx->LE()){
        qualifier.push(Q_LE);
      }else if(ctx->GTU()){
        qualifier.push(Q_GTU);
      }else if(ctx->LEU()){
        qualifier.push(Q_LEU);
      }else if(ctx->DOTADD()){
        qualifier.push(Q_DOTADD);
      }else if(ctx->GEU()){
        qualifier.push(Q_GEU);
      }else if(ctx->RZI()){
        qualifier.push(Q_RZI);
      }else if(ctx->DOTOR()){
        qualifier.push(Q_DOTOR);
      }else assert(0 && "some qualifier not recognized!\n");

      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterParams(ptxParser::ParamsContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitParams(ptxParser::ParamsContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterParam(ptxParser::ParamContext *ctx) override { 
      paramContext = new ParamContext();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitParam(ptxParser::ParamContext *ctx) override { 
      int digit_idx = 0;

      /* ID */
      paramContext->paramName = ctx->ID()->getText();

      /* align */
      if(ctx->ALIGN()){
        digit_idx++;
        paramContext->paramAlign = stoi(ctx->DIGITS(0)->getText());
      }else{
        paramContext->paramAlign = 0;
      }

      /* paramNum */
      if(ctx->LeftBracket()){
        paramContext->paramNum = stoi(ctx->DIGITS(digit_idx)->getText());
      }else{
        paramContext->paramNum = 1;
      }

      /* qualifier */
      paramContext->paramType = qualifier.front();
      qualifier.pop();

      /* end of parsing param */
      kernelContext->kernelParams.push_back(*paramContext);
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
      delete paramContext;
    }

    void enterCompoundStatement(ptxParser::CompoundStatementContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitCompoundStatement(ptxParser::CompoundStatementContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterStatements(ptxParser::StatementsContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitStatements(ptxParser::StatementsContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterStatement(ptxParser::StatementContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitStatement(ptxParser::StatementContext *ctx) override { 
      assert(op.size()==0);
      statementContext.statementType = statementType;
      statementContext.statement = statement;
      kernelContext->kernelStatements.push_back(statementContext);
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterRegStatement(ptxParser::RegStatementContext *ctx) override { 
      statement = new StatementContext::REG();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitRegStatement(ptxParser::RegStatementContext *ctx) override { 
      auto st = (StatementContext::REG*) statement;

      /* qualifier */
      while(qualifier.size()){
        st->regDataType.push_back(qualifier.front());
        qualifier.pop();
      }

      /* reg */
      assert(op.size());
      assert(op.front()->opType==O_REG);
      auto reg = *(OperandContext::REG*)op.front()->operand;
      st->regName = reg.regMajorName;
      op.pop();

      /* digits */
      if(ctx->DIGITS()){
        st->regNum = stoi(ctx->DIGITS()->getText());
      }else{
        st->regNum = 1;
      }

      /* end */
      statementType = S_REG;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterSharedStatement(ptxParser::SharedStatementContext *ctx) override { 
      statement = new StatementContext::SHARED();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitSharedStatement(ptxParser::SharedStatementContext *ctx) override { 
      auto st = (StatementContext::SHARED *)statement;

      /* align */
      st->sharedAlign = stoi(ctx->DIGITS(0)->getText());

      /* qualifier */
      while(qualifier.size()){
        st->sharedDataType.push_back(qualifier.front());
        qualifier.pop();
      }

      /* ID */
      st->sharedName = ctx->ID()->getText();

      /* size */
      if(ctx->DIGITS(1)){
        st->sharedSize = stoi(ctx->DIGITS(1)->getText());
      }else{
        st->sharedSize = 0;
      }

      /* end */
      statementType = S_SHARED;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterLocalStatement(ptxParser::LocalStatementContext *ctx) override { 
      statement = new StatementContext::LOCAL();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitLocalStatement(ptxParser::LocalStatementContext *ctx) override { 
      auto st = (StatementContext::LOCAL *)statement;

      /* align */
      st->localAlign = stoi(ctx->DIGITS(0)->getText());

      /* qualifier */
      while(qualifier.size()){
        st->localDataType.push_back(qualifier.front());
        qualifier.pop();
      }

      /* ID */
      st->localName = ctx->ID()->getText();

      /* size */
      st->localSize = stoi(ctx->DIGITS(1)->getText());

      /* end */
      statementType = S_LOCAL;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterDollorStatement(ptxParser::DollorStatementContext *ctx) override { 
      statement = new StatementContext::DOLLOR();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitDollorStatement(ptxParser::DollorStatementContext *ctx) override { 
      auto st = (StatementContext::DOLLOR *)statement;

      /* ID */
      st->dollorName = ctx->ID()->getText();

      /* end */
      statementType = S_DOLLOR;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterAtStatement(ptxParser::AtStatementContext *ctx) override { 
      statement = new StatementContext::AT();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitAtStatement(ptxParser::AtStatementContext *ctx) override { 
      auto st = (StatementContext::AT *)statement;

      /* reg */
      fetchOperand(st->atOp);

      /* ID */
      st->atLabelName = ctx->ID()->getText();

      /* end */
      statementType = S_AT;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterPragmaStatement(ptxParser::PragmaStatementContext *ctx) override { 
      statement = new StatementContext::PRAGMA();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitPragmaStatement(ptxParser::PragmaStatementContext *ctx) override { 
      auto st = (StatementContext::PRAGMA *)statement;

      /* prama string */
      st->pragmaString = ctx->STRING()->getText();

      /* end */
      statementType = S_PRAGMA;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterRetStatement(ptxParser::RetStatementContext *ctx) override { 
      statement = new StatementContext::RET();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitRetStatement(ptxParser::RetStatementContext *ctx) override { 
      auto st = (StatementContext::RET *)statement;
      /* end */
      statementType = S_RET;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterBarStatement(ptxParser::BarStatementContext *ctx) override { 
      statement = new StatementContext::BAR();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
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
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterBraStatement(ptxParser::BraStatementContext *ctx) override { 
      statement = new StatementContext::BRA();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitBraStatement(ptxParser::BraStatementContext *ctx) override {
      auto st = (StatementContext::BRA *)statement; 

      /* qualifier */
      if(qualifier.size()){
        st->braQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* ID */
      st->braLabel = ctx->ID()->getText();

      /* end */
      statementType = S_BRA;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterRcpStatement(ptxParser::RcpStatementContext *ctx) override { 
      statement = new StatementContext::RCP();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitRcpStatement(ptxParser::RcpStatementContext *ctx) override { 
      auto st = (StatementContext::RCP *)statement;
      
      /* qualifier */
      while(qualifier.size()){
        st->rcpQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op */
      for(int i=0;i<2;i++){
        fetchOperand(st->rcpOp[i]);
      }

      /* end */
      statementType = S_RCP;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterLdStatement(ptxParser::LdStatementContext *ctx) override { 
      statement = new StatementContext::LD();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitLdStatement(ptxParser::LdStatementContext *ctx) override {
      auto st = (StatementContext::LD *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->ldQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->ldOp[i]);
      }

      /* end */
      statementType = S_LD; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterMovStatement(ptxParser::MovStatementContext *ctx) override { 
      statement = new StatementContext::MOV();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitMovStatement(ptxParser::MovStatementContext *ctx) override { 
      auto st = (StatementContext::MOV *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->movQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->movOp[i]);
      }

      /* end */
      statementType = S_MOV;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterSetpStatement(ptxParser::SetpStatementContext *ctx) override {
      statement = new StatementContext::SETP(); 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitSetpStatement(ptxParser::SetpStatementContext *ctx) override { 
      auto st = (StatementContext::SETP *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->setpQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->setpOp[i]);
      }

      /* end */
      statementType = S_SETP;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterCvtaStatement(ptxParser::CvtaStatementContext *ctx) override { 
      statement = new StatementContext::CVTA();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitCvtaStatement(ptxParser::CvtaStatementContext *ctx) override {
      auto st = (StatementContext::CVTA *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->cvtaQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->cvtaOp[i]);
      }

      /* end */
      statementType = S_CVTA;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterCvtStatement(ptxParser::CvtStatementContext *ctx) override {
      statement = new StatementContext::CVT(); 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitCvtStatement(ptxParser::CvtStatementContext *ctx) override { 
      auto st = (StatementContext::CVT *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->cvtQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->cvtOp[i]);
      }

      /* end */
      statementType = S_CVT;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterMulStatement(ptxParser::MulStatementContext *ctx) override { 
      statement = new StatementContext::MUL();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitMulStatement(ptxParser::MulStatementContext *ctx) override { 
      auto st = (StatementContext::MUL *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->mulQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->mulOp[i]);
      }

      /* end */
      statementType = S_MUL;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterDivStatement(ptxParser::DivStatementContext *ctx) override { 
      statement = new StatementContext::DIV();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitDivStatement(ptxParser::DivStatementContext *ctx) override { 
      auto st = (StatementContext::DIV *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->divQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->divOp[i]);
      }

      /* end */
      statementType = S_DIV;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterSubStatement(ptxParser::SubStatementContext *ctx) override { 
      statement = new StatementContext::SUB();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitSubStatement(ptxParser::SubStatementContext *ctx) override { 
      auto st = (StatementContext::SUB *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->subQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->subOp[i]);
      }

      /* end */
      statementType = S_SUB;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterAddStatement(ptxParser::AddStatementContext *ctx) override { 
      statement = new StatementContext::ADD();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitAddStatement(ptxParser::AddStatementContext *ctx) override { 
      auto st = (StatementContext::ADD *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->addQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->addOp[i]);
      }

      /* end */
      statementType = S_ADD;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterShlStatement(ptxParser::ShlStatementContext *ctx) override { 
      statement = new StatementContext::SHL();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitShlStatement(ptxParser::ShlStatementContext *ctx) override { 
      auto st = (StatementContext::SHL *)statement;
      
      /* qualifier */
      while(qualifier.size()){
        st->shlQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->shlOp[i]);
      }

      /* end */
      statementType = S_SHL;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterShrStatement(ptxParser::ShrStatementContext *ctx) override { 
      statement = new StatementContext::SHR();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitShrStatement(ptxParser::ShrStatementContext *ctx) override { 
      auto st = (StatementContext::SHR *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->shrQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->shrOp[i]);
      }

      /* end */
      statementType = S_SHR;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterMaxStatement(ptxParser::MaxStatementContext *ctx) override { 
      statement = new StatementContext::MAX();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitMaxStatement(ptxParser::MaxStatementContext *ctx) override { 
      auto st = (StatementContext::MAX *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->maxQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->maxOp[i]);
      }

      /* end */
      statementType = S_MAX;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterMinStatement(ptxParser::MinStatementContext *ctx) override { 
      statement = new StatementContext::MIN();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitMinStatement(ptxParser::MinStatementContext *ctx) override { 
      auto st = (StatementContext::MIN *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->minQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->minOp[i]);
      }

      /* end */
      statementType = S_MIN;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterAndStatement(ptxParser::AndStatementContext *ctx) override { 
      statement = new StatementContext::AND();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitAndStatement(ptxParser::AndStatementContext *ctx) override { 
      auto st = (StatementContext::AND *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->andQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->andOp[i]);
      }

      /* end */
      statementType = S_AND;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterOrStatement(ptxParser::OrStatementContext *ctx) override { 
      statement = new StatementContext::OR();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitOrStatement(ptxParser::OrStatementContext *ctx) override { 
      auto st = (StatementContext::OR *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->orQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->orOp[i]);
      }

      /* end */
      statementType = S_OR;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterStStatement(ptxParser::StStatementContext *ctx) override { 
      statement = new StatementContext::ST();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitStStatement(ptxParser::StStatementContext *ctx) override { 
      auto st = (StatementContext::ST *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->stQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->stOp[i]);
      }

      /* end */
      statementType = S_ST;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterSelpStatement(ptxParser::SelpStatementContext *ctx) override { 
      statement = new StatementContext::SELP();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitSelpStatement(ptxParser::SelpStatementContext *ctx) override { 
      auto st = (StatementContext::SELP *)statement;
      
      /* qualifier */
      while(qualifier.size()){
        st->selpQualifier.push_back(qualifier.front());
        qualifier.pop();
      }
      
      /* op4 */
      for(int i=0;i<4;i++){
        fetchOperand(st->selpOp[i]);
      }
      
      /* end */
      statementType = S_SELP;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterMadStatement(ptxParser::MadStatementContext *ctx) override { 
      statement = new StatementContext::MAD();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitMadStatement(ptxParser::MadStatementContext *ctx) override { 
      auto st = (StatementContext::MAD *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->madQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op4 */
      for(int i=0;i<4;i++){
        fetchOperand(st->madOp[i]);
      }

      /* end */
      statementType = S_MAD;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterFmaStatement(ptxParser::FmaStatementContext *ctx) override { 
      statement = new StatementContext::FMA();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitFmaStatement(ptxParser::FmaStatementContext *ctx) override {
      auto st = (StatementContext::FMA *)statement; 

      /* qualifier */
      while(qualifier.size()){
        st->fmaQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op4 */
      for(int i=0;i<4;i++){
        fetchOperand(st->fmaOp[i]);
      }

      /* end */
      statementType = S_FMA;
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterWmmaStatement(ptxParser::WmmaStatementContext *ctx) override { 
      statement = new StatementContext::WMMA();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitWmmaStatement(ptxParser::WmmaStatementContext *ctx) override {
      auto st = (StatementContext::WMMA *)statement;

      /* wmmatype & op */
      if(ctx->LOAD()){
        st->wmmaType = WMMA_LOAD;
        for(int i=0;i<3;i++){
          fetchOperand(st->wmmaOp[i]);
        }
      }else if(ctx->STORE()){
        st->wmmaType = WMMA_STORE;
        for(int i=0;i<3;i++){
          fetchOperand(st->wmmaOp[i]);
        }
      }else if(ctx->WMMA()){
        st->wmmaType = WMMA_MMA;
        for(int i=0;i<4;i++){
          fetchOperand(st->wmmaOp[i]);
        }
      }

      /* qualifier */
      while(qualifier.size()){
        st->wmmaQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* end */
      statementType = S_WMMA; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterNegStatement(ptxParser::NegStatementContext *ctx) override { 
      statement = new StatementContext::NEG();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitNegStatement(ptxParser::NegStatementContext *ctx) override { 
      auto st = (StatementContext::NEG *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->negQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->negOp[i]);
      }

      /* end */
      statementType = S_NEG; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterNotStatement(ptxParser::NotStatementContext *ctx) override { 
      statement = new StatementContext::NOT();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitNotStatement(ptxParser::NotStatementContext *ctx) override { 
      auto st = (StatementContext::NOT *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->notQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->notOp[i]);
      }

      /* end */
      statementType = S_NOT; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterSqrtStatement(ptxParser::SqrtStatementContext *ctx) override { 
      statement = new StatementContext::SQRT();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitSqrtStatement(ptxParser::SqrtStatementContext *ctx) override { 
      auto st = (StatementContext::SQRT *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->sqrtQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->sqrtOp[i]);
      }

      /* end */
      statementType = S_SQRT; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterCosStatement(ptxParser::CosStatementContext *ctx) override { 
      statement = new StatementContext::COS();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitCosStatement(ptxParser::CosStatementContext *ctx) override { 
      auto st = (StatementContext::COS *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->cosQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->cosOp[i]);
      }

      /* end */
      statementType = S_COS; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterLg2Statement(ptxParser::Lg2StatementContext *ctx) override { 
      statement = new StatementContext::LG2();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitLg2Statement(ptxParser::Lg2StatementContext *ctx) override {
      auto st = (StatementContext::LG2 *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->lg2Qualifier.push_back(qualifier.front());
        qualifier.pop();
      } 

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->lg2Op[i]);
      }

      /* end */
      statementType = S_LG2; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterEx2Statement(ptxParser::Ex2StatementContext *ctx) override { 
      statement = new StatementContext::EX2();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitEx2Statement(ptxParser::Ex2StatementContext *ctx) override {
      auto st = (StatementContext::EX2 *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->ex2Qualifier.push_back(qualifier.front());
        qualifier.pop();
      } 

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->ex2Op[i]);
      }

      /* end */
      statementType = S_EX2; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterAtomStatement(ptxParser::AtomStatementContext *ctx) override { 
      statement = new StatementContext::ATOM();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitAtomStatement(ptxParser::AtomStatementContext *ctx) override { 
      auto st = (StatementContext::ATOM *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->atomQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 or op4 */
      if(ctx->operandFour()){
        for(int i=0;i<4;i++){
          fetchOperand(st->atomOp[i]);
        }
        st->operandNum = 4;
      }else if(ctx->operandThree()){
        for(int i=0;i<3;i++){
          fetchOperand(st->atomOp[i]);
        }
        st->operandNum = 3;
      }else assert(0);

      /* end */
      statementType = S_ATOM; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterXorStatement(ptxParser::XorStatementContext *ctx) override {
      statement = new StatementContext::XOR(); 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitXorStatement(ptxParser::XorStatementContext *ctx) override { 
      auto st = (StatementContext::XOR *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->xorQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op3 */
      for(int i=0;i<3;i++){
        fetchOperand(st->xorOp[i]);
      }

      /* end */
      statementType = S_XOR; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterAbsStatement(ptxParser::AbsStatementContext *ctx) override { 
      statement = new StatementContext::ABS();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitAbsStatement(ptxParser::AbsStatementContext *ctx) override { 
      auto st = (StatementContext::ABS *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->absQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->absOp[i]);
      }

      /* end */
      statementType = S_ABS; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterSinStatement(ptxParser::SinStatementContext *ctx) override { 
      statement = new StatementContext::SIN();
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitSinStatement(ptxParser::SinStatementContext *ctx) override { 
      auto st = (StatementContext::SIN *)statement;

      /* qualifier */
      while(qualifier.size()){
        st->sinQualifier.push_back(qualifier.front());
        qualifier.pop();
      }

      /* op2 */
      for(int i=0;i<2;i++){
        fetchOperand(st->sinOp[i]);
      }

      /* end */
      statementType = S_SIN; 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterReg(ptxParser::RegContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitReg(ptxParser::RegContext *ctx) override { 
      OperandContext *o = new OperandContext();
      OperandContext::REG *r = new OperandContext::REG();
      extractREG(ctx->ID(0)->getText(),r->regIdx,r->regMajorName);
      r->regMinorName = ctx->ID(1) ? ctx->ID(1)->getText() : "";
      o->operand = r;
      o->opType = O_REG;
      op.push(o); 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterVector(ptxParser::VectorContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitVector(ptxParser::VectorContext *ctx) override { 
      OperandContext *o = new OperandContext();
      OperandContext::VEC *v = new OperandContext::VEC();

      for(int i=0;i<ctx->regi().size();i++){
        OperandContext oc;
        oc.opType = O_REG;
        oc.operand = new OperandContext::REG();
        auto r = (OperandContext::REG*)oc.operand;
        extractREG(ctx->regi(i)->ID(0)->getText(),r->regIdx,r->regMajorName);
        r->regMinorName = ctx->regi(i)->ID(1) ? ctx->regi(i)->ID(1)->getText() : "";
        v->vec.push_back(oc);
      }
      o->operand = v;
      o->opType = O_VEC;
      op.push(o);
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterFetchAddress(ptxParser::FetchAddressContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitFetchAddress(ptxParser::FetchAddressContext *ctx) override { 
      OperandContext *o = new OperandContext();
      OperandContext::FA *fa = new OperandContext::FA();

      /* base */
      if(ctx->ID()){
        fa->ID = ctx->ID()->getText();
        fa->reg = nullptr;
      }else if(ctx->reg()){
        // assume base not require regMinorName
        fa->reg = new OperandContext();
        fetchOperand(*fa->reg);
      }else assert(0);

      /* minus */
      fa->ifMinus = ctx->MINUS() ? true : false;

      /* offset */
      if(ctx->DIGITS()){
        fa->offset = ctx->DIGITS()->getText();
      }else{
        fa->offset = "";
      }
      
      /* end */
      o->operand = fa;
      o->opType = O_FA;
      op.push(o);
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterImm(ptxParser::ImmContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitImm(ptxParser::ImmContext *ctx) override { 
      OperandContext *o = new OperandContext();
      OperandContext::IMM *imm = new OperandContext::IMM();

      imm->immVal = ctx->DIGITS()->getText();
      o->operand = imm;
      o->opType = O_IMM;
      op.push(o);
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }

    void enterVar(ptxParser::VarContext *ctx) override { 
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
    void exitVar(ptxParser::VarContext *ctx) override { 
      OperandContext *o = new OperandContext();
      OperandContext::VAR *var = new OperandContext::VAR();

      var->varName = ctx->ID()->getText();
      o->operand = var;
      o->opType = O_VAR;
      op.push(o);
      #ifdef LOG
      std::cout << __func__ << std::endl;
      #endif
    }
};


#endif