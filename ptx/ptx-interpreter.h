/**
 * @author gtyinstinct
 * ptx interpreter
 * TODO lots of BUG 
*/

#ifndef __PTX_INTERPRETER__
#define __PTX_INTERPRETER__

#include<driver_types.h>

#include "ptx-semantic.h"

class PtxInterpreter{
  public:
    PtxContext *ptxContext;
    KernelContext *kernelContext;
    void **kernelArgs;
    dim3 gridDim,blockDim;

    dim3 curGridIdx,curBlockIdx;


    class Symtable{
        public:
        Qualifier symType;
        int byteNum;
        int elementNum;
        std::string name;
        uint64_t val;
    };

    class Reg{
        public:
        Qualifier regType;
        int byteNum;
        int elementNum;
        std::string name;
        void *addr;
    };

    std::map<std::string,PtxInterpreter::Symtable*>name2Sym;
    std::map<std::string,PtxInterpreter::Reg*>name2Reg;

    void launchPtxInterpreter(PtxContext &ptx,std::string &kernel,void **args,
        dim3 &gridDim,dim3 &blockDim){

      // init 
      ptxContext = &ptx;
      kernelArgs = args;
      this->gridDim = gridDim;
      this->blockDim = blockDim;
      name2Sym.clear();
      name2Reg.clear();
      curBlockIdx.x = 0;
      curBlockIdx.y = 0;
      curBlockIdx.z = 0;
      curGridIdx.x = 0;
      curGridIdx.y = 0;
      curGridIdx.z = 0;
      // find KernelContext
      for(auto &e:ptx.ptxKernels){
        if(e.kernelName==kernel){
            kernelContext = &e;
        }
      }
      funcInterpreter();
    }

    void funcInterpreter(){
        // setup symbol for kernel args
        for(int i=0;i<kernelContext->kernelParams.size();i++){
            // temporily ignore align
            auto e = kernelContext->kernelParams[i];
            Symtable *s = new Symtable();
            s->name = e.paramName;
            s->elementNum = e.paramNum;
            s->symType = e.paramType;
            s->byteNum = Q2bits(e.paramType);
            s->val = (uint64_t)kernelArgs[i];
            name2Sym[s->name] = s;
        }
        
        // exe every thread
        while(1){
            // exe every inst
            std::printf("INTE: GridIdx(%d,%d,%d) BlockIdx(%d,%d,%d)\n",
                curGridIdx.x,curGridIdx.y,curGridIdx.z,
                curBlockIdx.x,curBlockIdx.y,curBlockIdx.z);
            for(auto &e:kernelContext->kernelStatements){
                exe_once(e);
            }
            curBlockIdx.x ++;
            if(curBlockIdx.x==blockDim.x){
                curBlockIdx.x = 0;
                curBlockIdx.y ++;
            }
            if(curBlockIdx.y==blockDim.y){
                curBlockIdx.y = 0;
                curBlockIdx.z ++;
            }
            if(curBlockIdx.z==blockDim.z){
                curBlockIdx.z = 0;
                curGridIdx.x ++;
            }
            if(curGridIdx.x==gridDim.x){
                curGridIdx.x = 0;
                curGridIdx.y ++;
            }
            if(curGridIdx.y==gridDim.y){
                curGridIdx.y = 0;
                curGridIdx.z ++;
            }
            if(curGridIdx.z==gridDim.z){
                break;
            }
        }
    }

    void exe_once(StatementContext &s){
        switch(s.statementType){
        case S_REG:{
            auto ss = (StatementContext::REG*)s.statement;
            if(name2Reg[ss->regName])return; // already alloc
            assert(ss->regDataType.size()==1);
            PtxInterpreter::Reg *reg = new PtxInterpreter::Reg();
            reg->regType = ss->regDataType.back();
            reg->name = ss->regName;
            reg->elementNum = ss->regNum;
            reg->byteNum = Q2bits(ss->regDataType.back());
            assert(reg->byteNum);
            reg->addr = calloc(reg->elementNum,reg->byteNum/8);
            name2Reg[reg->name] = reg;
            return;
        }
        case S_SHARED:{
            assert(0);
            return;
        }
        case S_LOCAL:{
            assert(0);
            return;
        }
        case S_DOLLOR:{
            assert(0);
            return;
        }
        case S_AT:{
            assert(0);
            return;
        }
        case S_PRAGMA:{
            assert(0);
            return;
        }
        case S_RET:{
            // do nothing
            return;
        }
        case S_BAR:{
            assert(0);
            return;
        }
        case S_BRA:{
            assert(0);
            return;
        }
        case S_RCP:{
            assert(0);
            return;
        }
        case S_LD:{
            auto ss = (StatementContext::LD*)s.statement;
            if(QvecHasQ(ss->ldQualifier,Q_PARAM)){
                // .param

                // process op0
                void *to;
                if(ss->ldOp[0].opType==O_REG){
                    to = getRegAddr((OperandContext::REG*)ss->ldOp[0].operand);
                }else assert(0);

                // process op1
                void *from;
                if(ss->ldOp[1].opType==O_FA){
                    auto fa = (OperandContext::FA*)ss->ldOp[1].operand;
                    void *base;
                    if(fa->reg){
                        base = getRegAddr((OperandContext::REG*)fa->reg->operand);
                    }else{
                        base = (void*)name2Sym[fa->ID]->val;
                    }
                    if(fa->ifMinus){ // TODO parse offset
                        from = (void*)(base + stoi(fa->offset,0,0));
                    }else{
                        from = (void*)(base - stoi(fa->offset,0,0));
                    }
                }else assert(0);

                // exe ld
                mov(from,to,getBits(ss->ldQualifier));
            }
            return;
        }
        case S_MOV:{
            auto ss = (StatementContext::MOV*)s.statement;
            
            // op0
            void *to;
            if(ss->movOp[0].opType==O_REG){
                to = getRegAddr((OperandContext::REG*)ss->movOp[0].operand);
            }else assert(0);

            // op1
            void *from;
            if(ss->movOp[1].opType==O_REG){
                auto op1 = (OperandContext::REG*)ss->movOp[1].operand;
                if(op1->regMinorName.size()==1){
                    static uint64_t t;
                    t = getInternReg(op1);
                    from = &t;
                }else{
                    from = getRegAddr(op1);
                }
            }else assert(0);

            // exe mov
            mov(from,to,getBits(ss->movQualifier));
            return;
        }
        case S_SETP:{
            assert(0);
            return;
        }
        case S_CVTA:{ // TODO origin purpose of CVTA?
            auto ss = (StatementContext::CVTA*)s.statement;

            // op0 
            void *to;
            if(ss->cvtaOp[0].opType==O_REG){
                to = getRegAddr((OperandContext::REG*)ss->cvtaOp[0].operand);
            }else assert(0);

            // op1
            void *from;
            if(ss->cvtaOp[1].opType==O_REG){
                from = getRegAddr((OperandContext::REG*)ss->cvtaOp[1].operand);
            }

            // exe cvta just mov
            mov(from,to,getBits(ss->cvtaQualifier));

            return;
        }
        case S_CVT:{
            assert(0);
            return;
        }
        case S_MUL:{
            auto ss = (StatementContext::MUL*)s.statement;

            // op0
            void *to;
            if(ss->mulOp[0].opType==O_REG){
                to = getRegAddr((OperandContext::REG*)ss->mulOp[0].operand);
            }else assert(0);

            // op1
            void *op1;
            if(ss->mulOp[1].opType==O_REG){
                op1 = getRegAddr((OperandContext::REG*)ss->mulOp[1].operand);
            }else assert(0);

            // op2
            void *op2;
            if(ss->mulOp[2].opType==O_IMM){
                static uint64_t t;
                auto imm = (OperandContext::IMM*)ss->mulOp[2].operand;
                t = stoi(imm->immVal,0,0); // TODO parse imm
                op2 = &t;
            }else assert(0);

            // exe mul
            mul(to,op1,op2,getBits(ss->mulQualifier));

            return;
        }
        case S_DIV:{
            assert(0);
            return;
        }
        case S_SUB:{
            assert(0);
            return;
        }
        case S_ADD:{
            auto ss = (StatementContext::ADD*)s.statement;

            // op0
            void *to;
            if(ss->addOp[0].opType==O_REG){
                to = getRegAddr((OperandContext::REG*)ss->addOp[0].operand);
            }else assert(0);

            // op1
            void *op1;
            if(ss->addOp[1].opType==O_REG){
                op1 = getRegAddr((OperandContext::REG*)ss->addOp[1].operand);
            }else assert(0);
            
            // op2 
            void *op2;
            if(ss->addOp[2].opType==O_REG){
                op2 = getRegAddr((OperandContext::REG*)ss->addOp[2].operand);
            }else assert(0);

            // exe add
            add(to,op1,op2,getBits(ss->addQualifier));

            return;
        }
        case S_SHL:{
            assert(0);
            return;
        }
        case S_SHR:{
            assert(0);
            return;
        }
        case S_MAX:{
            assert(0);
            return;
        }
        case S_MIN:{
            assert(0);
            return;
        }
        case S_AND:{
            assert(0);
            return;
        }
        case S_OR:{
            assert(0);
            return;
        }
        case S_ST:{
            auto ss = (StatementContext::ST*)s.statement;

            // op0
            void *to;
            assert(ss->stOp[0].opType==O_FA);
            auto fa = (OperandContext::FA*)ss->stOp[0].operand;
            void *base;
            if(fa->reg){
                base = getRegAddr((OperandContext::REG*)fa->reg->operand);
            }else{
                base = (void*)name2Sym[fa->ID]->val;
            }
            if(fa->ifMinus){
                to = base - stoi(fa->offset,0,0);
            }else{
                to = base + stoi(fa->offset,0,0);
            }
            to = (void*)*(uint64_t*)to; 

            // op1
            void *from;
            if(ss->stOp[1].opType==O_REG){
                from = getRegAddr((OperandContext::REG*)ss->stOp[1].operand);
            }else assert(0);

            // exe st
            mov(from,to,getBits(ss->stQualifier));

            return;
        }
        case S_SELP:{
            assert(0);
            return;
        }
        case S_MAD:{
            assert(0);
            return;
        }
        case S_FMA:{
            assert(0);
            return;
        }
        case S_WMMA:{
            assert(0);
            return;
        }
        case S_NEG:{
            assert(0);
            return;
        }
        case S_NOT:{
            assert(0);
            return;
        }
        case S_SQRT:{
            assert(0);
            return;
        }
        case S_COS:{
            assert(0);
            return;
        }
        case S_LG2:{
            assert(0);
            return;
        }
        case S_EX2:{
            assert(0);
            return;
        }
        case S_ATOM:{
            assert(0);
            return;
        }
        case S_XOR:{
            assert(0);
            return;
        }
        case S_ABS:{
            assert(0);
            return;
        }
        case S_SIN:{
            assert(0);
            return;
        }
        default: assert(0);
        }
    }

    int Q2bits(Qualifier &q){
        switch(q){
        case Q_S64:case Q_F64:case Q_B64:case Q_U64:return 8;
        case Q_S32:case Q_F32:case Q_B32:case Q_U32:return 4;
        case Q_S16:case Q_F16:case Q_B16:case Q_U16:return 2;
        case Q_S8:case Q_F8:case Q_B8:case Q_U8:return 1;
        case Q_PRED:return 1;
        default:return 0;
        }
    }

    bool QvecHasQ(std::vector<Qualifier>&qvec,Qualifier q){
        for(auto e:qvec){
            if(e==q)return true;
        }
        return false;
    }

    int getBits(std::vector<Qualifier>&q){
        int ret;
        for(auto e:q){
            if(ret=Q2bits(e))return ret;
        }
        return 0;
    }

    void *getRegAddr(OperandContext::REG *regContext){
        std::printf("INTE: access %s%d\n",
            regContext->regMajorName.c_str(),regContext->regIdx);
        auto reg = name2Reg[regContext->regMajorName];
        uint64_t offset = (regContext->regIdx-1) * reg->byteNum;
        return (void *)((uint64_t)reg->addr + offset);
    }

    uint64_t getInternReg(OperandContext::REG *regContext){
        if(regContext->regMajorName=="tid"){
            if(regContext->regMinorName=="x"){
                return curBlockIdx.x;
            }else if(regContext->regMinorName=="y"){
                return curBlockIdx.y;
            }else if(regContext->regMinorName=="z"){
                return curBlockIdx.z;
            }else assert(0);
        }else assert(0);
    }

    // TODO float or double mul
    void add(void *to,void *op1,void *op2,int len){
        switch(len){
        case 1:
        *(uint8_t*)to = *(uint8_t*)op1 + *(uint8_t*)op2;
        printf("INTE: add dest:%p op1:%d op2:%d\n",to,*(uint8_t*)op1,*(uint8_t*)op2);
        return;
        case 2:
        *(uint16_t*)to = *(uint16_t*)op1 + *(uint16_t*)op2;
        printf("INTE: add dest:%p op1:%d op2:%d\n",to,*(uint16_t*)op1,*(uint16_t*)op2);
        return;
        case 4:
        *(uint32_t*)to = *(uint32_t*)op1 + *(uint32_t*)op2;
        printf("INTE: add dest:%p op1:%d op2:%d\n",to,*(uint32_t*)op1,*(uint32_t*)op2);
        return;
        case 8:
        *(uint64_t*)to = *(uint64_t*)op1 + *(uint64_t*)op2;
        printf("INTE: add dest:%p op1:0x%lx op2:0x%lx\n",to,*(uint64_t*)op1,*(uint64_t*)op2);
        return;
        }
    }

    // TODO float or double mul
    void mul(void *to,void *op1,void *op2,int len){
        switch(len){
        case 1:
        *(uint8_t*)to = *(uint8_t*)op1 * *(uint8_t*)op2;
        printf("INTE: mul dest:%p op1:%d op2:%d\n",to,*(uint8_t*)op1,*(uint8_t*)op2);
        return;
        case 2:
        *(uint16_t*)to = *(uint16_t*)op1 * *(uint16_t*)op2;
        printf("INTE: mul dest:%p op1:%d op2:%d\n",to,*(uint16_t*)op1,*(uint16_t*)op2);
        return;
        case 4:
        *(uint32_t*)to = *(uint32_t*)op1 * *(uint32_t*)op2;
        printf("INTE: mul dest:%p op1:%d op2:%d\n",to,*(uint32_t*)op1,*(uint32_t*)op2);
        return;
        case 8:
        *(uint64_t*)to = *(uint64_t*)op1 * *(uint64_t*)op2;
        printf("INTE: mul dest:%p op1:%d op2:%d\n",to,*(uint64_t*)op1,*(uint64_t*)op2);
        return;
        }
    }

    void mov(void *from,void *to,int len){
        switch(len){
        case 1:
        *(uint8_t*)to = *(uint8_t*)from;
        std::printf("INTE: mov %p to %p data:%x\n",from,to,*(uint8_t*)from);
        return;
        case 2:
        *(uint16_t*)to = *(uint16_t*)from;
        std::printf("INTE: mov %p to %p data:%x\n",from,to,*(uint16_t*)from);
        return;
        case 4:
        *(uint32_t*)to = *(uint32_t*)from;
        std::printf("INTE: mov %p to %p data:%x\n",from,to,*(uint32_t*)from);
        return;
        case 8:
        *(uint64_t*)to = *(uint64_t*)from;
        std::printf("INTE: mov %p to %p data:%lx\n",from,to,*(uint64_t*)from);
        return;
        default:assert(0);
        }
    }

};
#endif