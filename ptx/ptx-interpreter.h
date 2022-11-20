/**
 * @author gtyinstinct
 * ptx interpreter
 * lots of BUG 
*/

#ifndef __PTX_INTERPRETER__
#define __PTX_INTERPRETER__

#include<driver_types.h>

#include "ptx-semantic.h"

enum DTYPE{
    DFLOAT,
    DINT,
    DNONE
};

class IMM{
    public:
    Qualifier type;
    union Data{
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        float f32;
        double f64;
    };
    Data data;
};


class PtxInterpreter{
  public:
    PtxContext *ptxContext;
    KernelContext *kernelContext;
    void **kernelArgs;
    dim3 gridDim,blockDim;

    dim3 curGridIdx,curBlockIdx;

    int pc;
    bool exit=0;

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
    std::map<std::string,int>name2pc;

    std::queue<IMM*> imm; // TODO fix memory leak

    void launchPtxInterpreter(PtxContext &ptx,std::string &kernel,void **args,
        dim3 &gridDim,dim3 &blockDim){

      // init 
      ptxContext = &ptx;
      kernelArgs = args;
      this->gridDim = gridDim;
      this->blockDim = blockDim;
      name2Sym.clear();
      name2Reg.clear();
      name2pc.clear();
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
            s->byteNum = Q2bytes(e.paramType);
            s->val = (uint64_t)kernelArgs[i];
            name2Sym[s->name] = s;
        }

        // setup label to jump
        for(int i=0;i<kernelContext->kernelStatements.size();i++){
            auto e = kernelContext->kernelStatements[i];
            if(e.statementType==S_DOLLOR){
                auto s = (StatementContext::DOLLOR*)e.statement;
                name2pc[s->dollorName] = i;
            }
        }
        
        // exe every thread
        while(1){
            // exe every inst
            pc = -1;
            exit = 0;
            while(!exit){
                pc++;
                assert(pc>=0 && pc<kernelContext->kernelStatements.size());
                auto e = kernelContext->kernelStatements[pc];
                #ifdef DEBUGINTE
                getchar();
                std::printf("\nINTE: GridIdx(%d,%d,%d) BlockIdx(%d,%d,%d)\n",
                    curGridIdx.x,curGridIdx.y,curGridIdx.z,
                    curBlockIdx.x,curBlockIdx.y,curBlockIdx.z);
                #endif
                exe_once(e);
                #ifdef DEBUGINTE
                std::printf("PC:%d\n",pc);
                int toti = 0;
                for(auto &e:name2Reg){
                    auto ee = e.second;
                    void *base = ee->addr;
                    for(int i=0;i<ee->elementNum;i++,toti++){
                        switch(ee->name[0]){
                        case 'r':
                        switch(ee->byteNum){
                        case 4:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                            *(uint32_t*)((uint64_t)base+i*ee->byteNum));break;
                        case 8:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                            *(uint64_t*)((uint64_t)base+i*ee->byteNum));break;
                        }
                        break;
                        case 'f':
                        switch(ee->byteNum){
                        case 4:std::printf("%5s%-3d:%-16f ",ee->name.c_str(),i,
                            *(float*)((uint64_t)base+i*ee->byteNum));break;
                        case 8:std::printf("%5s%-3d:%-16lf ",ee->name.c_str(),i,
                            *(double*)((uint64_t)base+i*ee->byteNum));break;
                        }
                        break;
                        case 'p':
                        std::printf("%5s%-3d:%-16d ",ee->name.c_str(),i,
                            *(uint8_t*)((uint64_t)base+i*ee->byteNum));break;
                        default:assert(0);
                        }
                        if((toti+1)%4==0)std::printf("\n");
                    }
                }
                std::printf("\n");
                #endif
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
        //#ifdef LOGINTE
        std::printf("INTE: %s\n",S2s(s.statementType).c_str());
        //#endif
        switch(s.statementType){
        case S_REG:{
            auto ss = (StatementContext::REG*)s.statement;
            if(name2Reg[ss->regName]){
                auto reg = name2Reg[ss->regName];
                memset(reg->addr,0,reg->elementNum*reg->byteNum);
                return; // already alloc
            }
            assert(ss->regDataType.size()==1);
            PtxInterpreter::Reg *reg = new PtxInterpreter::Reg();
            reg->regType = ss->regDataType.back();
            reg->name = ss->regName;
            reg->elementNum = ss->regNum;
            reg->byteNum = Q2bytes(ss->regDataType.back());
            assert(reg->byteNum);
            reg->addr = calloc(reg->elementNum,reg->byteNum);
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
            // do nothing
            return;
        }
        case S_AT:{
            auto ss = (StatementContext::AT*)s.statement;

            // pred
            std::vector<Qualifier>t;
            void *pred = getOperandAddr(ss->atOp,t);

            // exe at
            at(pred,ss->atLabelName);

            return;
        }
        case S_PRAGMA:{
            assert(0);
            return;
        }
        case S_RET:{
            exit = 1;
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
                void *to = getOperandAddr(ss->ldOp[0],ss->ldQualifier);

                // process op1
                void *from = getOperandAddr(ss->ldOp[1],ss->ldQualifier);

                // exe ld
                mov(from,to,ss->ldQualifier);
            }else assert(0);
            return;
        }
        case S_MOV:{
            auto ss = (StatementContext::MOV*)s.statement;
            
            // op0
            void *to = getOperandAddr(ss->movOp[0],ss->movQualifier);

            // op1
            void *from = getOperandAddr(ss->movOp[1],ss->movQualifier);

            // exe mov
            mov(from,to,ss->movQualifier);
            return;
        }
        case S_SETP:{
            auto ss = (StatementContext::SETP*)s.statement;

            // op0
            void *to = getOperandAddr(ss->setpOp[0],ss->setpQualifier);

            // op1
            void *op0 = getOperandAddr(ss->setpOp[1],ss->setpQualifier);

            // op2
            void *op1 = getOperandAddr(ss->setpOp[2],ss->setpQualifier);

            // exe setp
            setp(to,op0,op1,ss->setpQualifier);

            return;
        }
        case S_CVTA:{ // TODO origin purpose of CVTA?
            auto ss = (StatementContext::CVTA*)s.statement;

            // op0 
            void *to = getOperandAddr(ss->cvtaOp[0],ss->cvtaQualifier);

            // op1
            void *from = getOperandAddr(ss->cvtaOp[1],ss->cvtaQualifier);

            // exe cvta just mov
            mov(from,to,ss->cvtaQualifier);

            return;
        }
        case S_CVT:{
            auto ss = (StatementContext::CVT*)s.statement;

            // op0
            void *to = getOperandAddr(ss->cvtOp[0],ss->cvtQualifier);

            // op1
            void *from = getOperandAddr(ss->cvtOp[1],ss->cvtQualifier);

            cvt(to,from,ss->cvtQualifier);
            return;
        }
        case S_MUL:{
            auto ss = (StatementContext::MUL*)s.statement;

            // op0
            void *to = getOperandAddr(ss->mulOp[0],ss->mulQualifier);

            // op1
            void *op1 = getOperandAddr(ss->mulOp[1],ss->mulQualifier);

            // op2
            void *op2 = getOperandAddr(ss->mulOp[2],ss->mulQualifier);

            // exe mul
            mul(to,op1,op2,ss->mulQualifier);

            return;
        }
        case S_DIV:{
            assert(0);
            return;
        }
        case S_SUB:{
            auto ss = (StatementContext::SUB*)s.statement;

            // op0
            void *to = getOperandAddr(ss->subOp[0],ss->subQualifier);

            // op1
            void *op1 = getOperandAddr(ss->subOp[1],ss->subQualifier);
            
            // op2 
            void *op2 = getOperandAddr(ss->subOp[2],ss->subQualifier);

            // exe add
            sub(to,op1,op2,ss->subQualifier);

            return;
        }
        case S_ADD:{
            auto ss = (StatementContext::ADD*)s.statement;

            // op0
            void *to = getOperandAddr(ss->addOp[0],ss->addQualifier);

            // op1
            void *op1 = getOperandAddr(ss->addOp[1],ss->addQualifier);
            
            // op2 
            void *op2 = getOperandAddr(ss->addOp[2],ss->addQualifier);

            // exe add
            add(to,op1,op2,ss->addQualifier);

            return;
        }
        case S_SHL:{
            auto ss = (StatementContext::SHL*)s.statement;

            // op0
            void *to = getOperandAddr(ss->shlOp[0],ss->shlQualifier);

            // op1
            void *op0 = getOperandAddr(ss->shlOp[1],ss->shlQualifier);

            // op2
            std::vector<Qualifier>tq;
            tq.push_back(Q_U32);
            void *op1 = getOperandAddr(ss->shlOp[2],tq);

            // exe shl
            shl(to,op0,op1,ss->shlQualifier);

            return;
        }
        case S_SHR:{
            auto ss = (StatementContext::SHR*)s.statement;

            // op0
            void *to = getOperandAddr(ss->shrOp[0],ss->shrQualifier);

            // op1
            void *op0 = getOperandAddr(ss->shrOp[1],ss->shrQualifier);

            // op2
            std::vector<Qualifier>tq;
            tq.push_back(Q_U32);
            void *op1 = getOperandAddr(ss->shrOp[2],tq);

            // exe shr
            shr(to,op0,op1,ss->shrQualifier);
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
            void *to = getOperandAddr(ss->stOp[0],ss->stQualifier);
            to = (void*)*(uint64_t*)to;

            // op1
            void *from = getOperandAddr(ss->stOp[1],ss->stQualifier);

            // exe st
            mov(from,to,ss->stQualifier);

            return;
        }
        case S_SELP:{
            auto ss = (StatementContext::ST*)s.statement;

            // op0
            void *to = getOperandAddr(ss->stOp[0],ss->stQualifier);

            // op1
            void *op0 = getOperandAddr(ss->stOp[1],ss->stQualifier);

            // op2
            void *op1 = getOperandAddr(ss->stOp[2],ss->stQualifier);

            // op3
            void *pred = getOperandAddr(ss->stOp[3],ss->stQualifier);

            // exe selp
            selp(to,op0,op1,pred,ss->stQualifier);

            return;
        }
        case S_MAD:{
            auto ss = (StatementContext::MAD*)s.statement;

            // op0 
            void *to = getOperandAddr(ss->madOp[0],ss->madQualifier);
            
            // op1
            void *op0 = getOperandAddr(ss->madOp[1],ss->madQualifier);

            // op2 
            void *op1 = getOperandAddr(ss->madOp[2],ss->madQualifier);

            // op3
            void *op2 = getOperandAddr(ss->madOp[3],ss->madQualifier);

            // exe mad
            void *t = malloc(16);

            mul(t,op0,op1,ss->madQualifier);

            add(to,t,op2,ss->madQualifier);

            free(t);
            return;
        }
        case S_FMA:{
            auto ss = (StatementContext::FMA*)s.statement;

            // op0
            void *to = getOperandAddr(ss->fmaOp[0],ss->fmaQualifier);

            // op1
            void *op0 = getOperandAddr(ss->fmaOp[1],ss->fmaQualifier);

            // op2
            void *op1 = getOperandAddr(ss->fmaOp[2],ss->fmaQualifier);

            // op3
            void *op2 = getOperandAddr(ss->fmaOp[3],ss->fmaQualifier);

            void *t = malloc(8);


            // exe fma
            mul(t,op0,op1,ss->fmaQualifier);

            add(to,t,op2,ss->fmaQualifier);

            free(t);
            return;
        }
        case S_WMMA:{
            assert(0);
            return;
        }
        case S_NEG:{
            auto ss = (StatementContext::NEG*)s.statement;

            // op0
            void *to = getOperandAddr(ss->negOp[0],ss->negQualifier);

            // op1
            void *op0 = getOperandAddr(ss->negOp[1],ss->negQualifier);

            // exe neg
            neg(to,op0,ss->negQualifier);

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
        case S_REM:{
            auto ss = (StatementContext::REM*)s.statement;

            // op0
            void *to = getOperandAddr(ss->remOp[0],ss->remQualifier);

            // op1
            void *op0 = getOperandAddr(ss->remOp[1],ss->remQualifier);

            // op2
            void *op1 = getOperandAddr(ss->remOp[2],ss->remQualifier);

            // exe rem
            rem(to,op0,op1,ss->remQualifier);

            return;
        }
        default: assert(0);
        }
    }

    // helper function

    void setIMM(std::string s,Qualifier q){
        IMM *t_imm = new IMM();
        t_imm->type = q;
        switch(q){
        case Q_S64:case Q_U64:case Q_B64:
        t_imm->data.u64 = stol(s,0,0);
        break;
        case Q_S32:case Q_U32:case Q_B32:
        t_imm->data.u32 = stoi(s,0,0);
        break;
        case Q_S16:case Q_U16:case Q_B16:
        t_imm->data.u16 = stoi(s,0,0);
        break;
        case Q_S8:case Q_U8:case Q_B8:
        t_imm->data.u8 = stoi(s,0,0);
        break;
        case Q_F64:
        assert(s.size()==18&&(s[1]=='d'||s[1]=='D'));
        s[1] = 'x';
        *(uint64_t*)&(t_imm->data.f64) = stoull(s,0,0);
        break;
        case Q_F32:
        assert(s.size()==10&&(s[1]=='f'||s[1]=='F'));
        s[1] = 'x';
        *(uint32_t*)&(t_imm->data.f32) = stoi(s,0,0);
        break;
        default:assert(0);
        }
        this->imm.push(t_imm);
    }

    int Q2bytes(Qualifier &q){
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

    Qualifier getDataType(std::vector<Qualifier>&q){
        for(auto e:q){
            if(Q2bytes(e))return e;
        }
        assert(0);
    }

    DTYPE getDType(std::vector<Qualifier>&q){
        for(auto e:q){
            switch(e){
            case Q_F64:case Q_F32:case Q_F16:case Q_F8: return DFLOAT;
            case Q_S64:case Q_B64:case Q_U64:
            case Q_S32:case Q_B32:case Q_U32:
            case Q_S16:case Q_B16:case Q_U16:
            case Q_S8:case Q_B8:case Q_U8:return DINT;
            }
        }
        assert(0);
    }

    DTYPE getDType(Qualifier q){
        switch(q){
        case Q_F64:case Q_F32:case Q_F16:case Q_F8: return DFLOAT;
        case Q_S64:case Q_B64:case Q_U64:
        case Q_S32:case Q_B32:case Q_U32:
        case Q_S16:case Q_B16:case Q_U16:
        case Q_S8:case Q_B8:case Q_U8:return DINT;
        }
        return DNONE;
    }

    int getBits(std::vector<Qualifier>&q){
        int ret;
        for(auto e:q){
            if(ret=Q2bytes(e))return ret;
        }
        return 0;
    }

    int getBits(Qualifier q){
        return Q2bytes(q);
    }

    void *getOperandAddr(OperandContext &op,std::vector<Qualifier>&q){
        if(op.opType==O_REG){
            return getRegAddr((OperandContext::REG*)op.operand);
        }else if(op.opType==O_FA){
            return getFaAddr((OperandContext::FA*)op.operand);
        }else if(op.opType==O_IMM){
            setIMM(((OperandContext::IMM*)op.operand)->immVal,
                    getDataType(q));
            void *t = &(this->imm.front()->data);
            this->imm.pop();
            return t;
        }else assert(0);
    }

    void *getRegAddr(OperandContext::REG *regContext){
        if(regContext->regMinorName.size()==1){
            #ifdef LOGINTE
            std::printf("INTE: access %s.%s\n",
                regContext->regMajorName.c_str(),regContext->regMinorName.c_str());
            #endif
            static uint64_t t;
            t = getSpReg(regContext);
            return &t;
        }else{
            #ifdef LOGINTE
            std::printf("INTE: access %s%d\n",
                regContext->regMajorName.c_str(),regContext->regIdx);
            #endif
            auto reg = name2Reg[regContext->regMajorName];
            uint64_t offset = regContext->regIdx * reg->byteNum;
            return (void *)((uint64_t)reg->addr + offset);
        }
    }

    void *getFaAddr(OperandContext::FA *fa){
        void *ret;
        if(fa->reg){
            ret = getRegAddr((OperandContext::REG*)fa->reg->operand);
        }else{
            ret = (void*)name2Sym[fa->ID]->val;
        }
        if(fa->offset.size()!=0){
            setIMM(fa->offset,Q_U64);
            IMM *t = this->imm.front();
            this->imm.pop();
            if(fa->ifMinus){ 
                ret = (void*)((uint64_t)ret + t->data.u64);
            }else{
                ret = (void*)((uint64_t)ret - t->data.u64);
            }
            free(t);
        }
        return ret;
    }

    Qualifier getMulQ(std::vector<Qualifier>&q){
        for(auto e:q){
            if(e==Q_WIDE||e==Q_HI||e==Q_LO){
                return e;
            }
        }
        assert(0);
    }

    uint64_t getSpReg(OperandContext::REG *regContext){
        if(regContext->regMajorName=="tid"){
            if(regContext->regMinorName=="x"){
                return curBlockIdx.x;
            }else if(regContext->regMinorName=="y"){
                return curBlockIdx.y;
            }else if(regContext->regMinorName=="z"){
                return curBlockIdx.z;
            }else assert(0);
        }else if(regContext->regMajorName=="ctaid"){
            if(regContext->regMinorName=="x"){
                return curGridIdx.x;
            }else if(regContext->regMinorName=="y"){
                return curGridIdx.y;
            }else if(regContext->regMinorName=="z"){
                return curGridIdx.z;
            }else assert(0);
        }else if(regContext->regMajorName=="ntid"){
            if(regContext->regMinorName=="x"){
                return blockDim.x;
            }else if(regContext->regMinorName=="y"){
                return blockDim.y;
            }else if(regContext->regMinorName=="z"){
                return blockDim.z;
            }else assert(0);
        }else if(regContext->regMajorName=="nctaid"){
            if(regContext->regMinorName=="x"){
                return gridDim.x;
            }else if(regContext->regMinorName=="y"){
                return gridDim.y;
            }else if(regContext->regMinorName=="z"){
                return gridDim.z;
            }else assert(0);
        }else assert(0);
    }

    void at(void *pred,std::string &label){
        if(*(uint8_t*)pred){
            pc = name2pc[label];
            assert(pc!=0);
        }
    }

    template<typename T>
    void _rem(void *to,void *op0,void *op1){
        *(T*)to = *(T*)op0 % *(T*)op1 ;
    }

    void rem(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
        int len = getBits(q);
        Qualifier datatype = getDataType(q);
        switch(len){
        case 1: {
            if(Signed(datatype))
                _rem<int8_t>(to,op0,op1);
            else
                _rem<uint8_t>(to,op0,op1);
            return;
        }
        case 2: {
            if(Signed(datatype))
                _rem<int16_t>(to,op0,op1);
            else
                _rem<uint16_t>(to,op0,op1);
            return;
        }
        case 4: {
            if(Signed(datatype))
                _rem<int32_t>(to,op0,op1);
            else
                _rem<uint32_t>(to,op0,op1);
            return;
        }
        case 8: {
            if(Signed(datatype))
                _rem<int64_t>(to,op0,op1);
            else
                _rem<uint64_t>(to,op0,op1);
            return;
        }
        default: assert(0);
        }
    }

    template<typename T>
    void _selp(void *to,void *op0,void *op1,void *pred){
        *(T*)to = *(uint8_t*)pred ? *(T*)op0 : *(T*)op1 ;
    }

    void selp(void *to,void *op0,void *op1,void *pred,std::vector<Qualifier>&q){
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        switch(len){
        case 1:
        assert(dtype==DINT);
        _selp<uint8_t>(to,op0,op1,pred);
        return;
        case 2:
        assert(dtype==DINT);
        _selp<uint16_t>(to,op0,op1,pred);
        return;
        case 4:
        switch(dtype){
            case DINT: _selp<uint32_t>(to,op0,op1,pred);return;
            case DFLOAT: _selp<float>(to,op0,op1,pred);return;
            default: assert(0);
        }
        return;
        case 8:
        switch(dtype){
            case DINT: _selp<uint64_t>(to,op0,op1,pred);return;
            case DFLOAT: _selp<double>(to,op0,op1,pred);return;
            default: assert(0);
        }
        default: assert(0);
        }
    }

    template<typename T>
    void _neg(void *to,void *op0){
        *(T*)to = - *(T*)op0;
    }

    void neg(void *to,void *op0,std::vector<Qualifier>&q){
        Qualifier datatype = getDataType(q);
        switch(datatype){
        case Q_S16:
        _neg<int16_t>(to,op0);
        return;
        case Q_S32:
        _neg<int32_t>(to,op0);
        return;
        case Q_S64:
        _neg<int64_t>(to,op0);
        return;
        default:assert(0);
        }
    }

    Qualifier getCMPOP(std::vector<Qualifier>&q){
        for(auto e:q){
            switch(e){
            case Q_EQ:case Q_NE:case Q_LT:case Q_LE:case Q_GT:
            case Q_GE:case Q_LO:case Q_HI:
            return e;
            }
        }
        assert(0);
    }

    template<typename T>
    void _setp_eq(void *to,void *op0,void *op1){
        *(uint8_t*)to = *(T*)op0 == *(T*)op1;
    }

    template<typename T>
    void _setp_lt(void *to,void *op0,void *op1){
        *(uint8_t*)to = *(T*)op0 < *(T*)op1;
    }

    template<typename T>
    void _setp_le(void *to,void *op0,void *op1){
        *(uint8_t*)to = *(T*)op0 <= *(T*)op1;
    }

    void setp(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
        Qualifier cmpOp = getCMPOP(q);
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        Qualifier datatype = getDataType(q);
        switch(cmpOp){
        case Q_EQ:{ 
            switch(len){
            case 1: {
                assert(dtype==DINT);
                _setp_eq<uint8_t>(to,op0,op1);
                return;
            }
            case 2:{
                assert(dtype==DINT);
                _setp_eq<uint16_t>(to,op0,op1);
                return;
            }
            case 4:{
                assert(dtype==DINT);
                _setp_eq<uint32_t>(to,op0,op1);
                return;
            }
            case 8:{
                assert(dtype==DINT);
                _setp_eq<uint64_t>(to,op0,op1);
                return;
            }
            default:assert(0);
            }
            return;
        }
        case Q_LT:{
            switch(len){
            case 1: {
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_lt<int8_t>(to,op0,op1);
                else 
                    _setp_lt<uint8_t>(to,op0,op1);
                return;
            }
            case 2:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_lt<int16_t>(to,op0,op1);
                else 
                    _setp_lt<uint16_t>(to,op0,op1);
                return;
            }
            case 4:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_lt<int32_t>(to,op0,op1);
                else
                    _setp_lt<uint32_t>(to,op0,op1);
                return;
            }
            case 8:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_lt<int64_t>(to,op0,op1);
                else
                    _setp_lt<uint64_t>(to,op0,op1);
                return;
            }
            default:assert(0);
            }
            return;
        }
        case Q_LE:{
            switch(len){
            case 1: {
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_le<int8_t>(to,op0,op1);
                else 
                    _setp_le<uint8_t>(to,op0,op1);
                return;
            }
            case 2:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_le<int16_t>(to,op0,op1);
                else 
                    _setp_le<uint16_t>(to,op0,op1);
                return;
            }
            case 4:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_le<int32_t>(to,op0,op1);
                else
                    _setp_le<uint32_t>(to,op0,op1);
                return;
            }
            case 8:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _setp_le<int64_t>(to,op0,op1);
                else
                    _setp_le<uint64_t>(to,op0,op1);
                return;
            }
            default:assert(0);
            }
            return;
        }
        default:assert(0);
        }
    }

    bool Signed(Qualifier q){
        switch(q){
        case Q_S64:case Q_S32:case Q_S16:case Q_S8:return 1;
        default:return 0; 
        }
    }

    template<typename T>
    void _shr(void *to,void *op0,void *op1){
        *(T*)to = *(T*)op0 >> *(uint32_t*)op1 ;
    }

    void shr(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
        int len = getBits(q);
        Qualifier datatype = getDataType(q);
        switch(len){
        case 1: {
            if(Signed(datatype))
                _shr<int8_t>(to,op0,op1);
            else
                _shr<uint8_t>(to,op0,op1);
            return;
        }
        case 2: {
            if(Signed(datatype))
                _shr<int16_t>(to,op0,op1);
            else
                _shr<uint16_t>(to,op0,op1);
            return;
        }
        case 4: {
            if(Signed(datatype))
                _shr<int32_t>(to,op0,op1);
            else
                _shr<uint32_t>(to,op0,op1);
            return;
        }
        case 8: {
            if(Signed(datatype))
                _shr<int64_t>(to,op0,op1);
            else
                _shr<uint64_t>(to,op0,op1);
            return;
        }
        default: assert(0);
        }
    }

    template<typename T>
    void _shl(void *to,void *op0,void *op1){
        *(T*)to = *(T*)op0 << *(uint32_t*)op1 ;
    }

    void shl(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
        int len = getBits(q);
        switch(len){
        case 1: _shl<uint8_t>(to,op0,op1);return;
        case 2: _shl<uint16_t>(to,op0,op1);return;
        case 4: _shl<uint32_t>(to,op0,op1);return;
        case 8: _shl<uint64_t>(to,op0,op1);return;
        default: assert(0);
        }
    }

    template<typename T1,typename T2>
    void __cvt(void *to,void *from){
        *(T1*)to = *(T2*)from;
    }

    template<typename T>
    void _cvt(void *to,void *from,int bitnum,DTYPE dtype){
        switch(bitnum){
        case 8: {
            if(dtype==DINT)
                __cvt<T,uint64_t>(to,from);
            else if(dtype==DFLOAT)
                __cvt<T,double>(to,from);
            else assert(0);
            return;
        }
        case 4: {
            if(dtype==DINT)
                __cvt<T,uint32_t>(to,from);
            else if(dtype==DFLOAT)
                __cvt<T,float>(to,from);
            else assert(0);
            return;
        }
        case 2:{
            assert(dtype==DINT);
            __cvt<T,uint16_t>(to,from);
            return;
        }
        case 1:{
            assert(dtype==DINT);
            __cvt<T,uint8_t>(to,from);
            return;
        }
        default: assert(0);
        }
    }

    void cvt(void *to,void *from,std::vector<Qualifier>&q){
        int bitnum[2],idx=0;
        for(auto e:q){
            if(getBits(e))bitnum[idx++] = getBits(e);
            if(idx==2)break;
        }
        assert(idx==2);
        DTYPE dtype[2];
        idx = 0;
        for(auto e:q){
            if(getDType(e)!=DNONE)dtype[idx++] = getDType(e);
            if(idx==2)break;
        }
        assert(idx==2);
        switch(bitnum[0]){
            case 8: {
                if(dtype[0]==DINT)
                    _cvt<uint64_t>(to,from,bitnum[1],dtype[1]);
                else if(dtype[0]==DFLOAT)
                    _cvt<double>(to,from,bitnum[1],dtype[1]);
                else assert(0);
                return;
            }
            case 4: {
                if(dtype[0]==DINT)
                    _cvt<uint32_t>(to,from,bitnum[1],dtype[1]);
                else if(dtype[0]==DFLOAT)
                    _cvt<float>(to,from,bitnum[1],dtype[1]);
                else assert(0);
                return;
            }
            case 2:{
                assert(dtype[0]==DINT);
                _cvt<uint16_t>(to,from,bitnum[1],dtype[1]);
                return;
            }
            case 1:{
                assert(dtype[0]==DINT);
                _cvt<uint8_t>(to,from,bitnum[1],dtype[1]);
                return;
            }
            default: assert(0);
        }
        
    }

    template<typename T>
    void _sub(void *to,void *op1,void *op2){
        *(T*)to = *(T*)op1 - *(T*)op2;
    }

    void sub(void *to,void *op1,void *op2,std::vector<Qualifier>&q){
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        switch(len){
        case 1:
        assert(dtype==DINT);
        _sub<uint8_t>(to,op1,op2);
        return;
        case 2:
        assert(dtype==DINT);
        _sub<uint16_t>(to,op1,op2);
        return;
        case 4:
        switch(dtype){
            case DINT: _sub<uint32_t>(to,op1,op2);return;
            case DFLOAT: _sub<float>(to,op1,op2);return;
            default: assert(0);
        }
        return;
        case 8:
        switch(dtype){
            case DINT: _sub<uint64_t>(to,op1,op2);return;
            case DFLOAT: _sub<double>(to,op1,op2);return;
            default: assert(0);
        }
        default: assert(0);
        }
    }

    template<typename T>
    void _add(void *to,void *op1,void *op2){
        *(T*)to = *(T*)op1 + *(T*)op2;
    }

    void add(void *to,void *op1,void *op2,std::vector<Qualifier>&q){
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        switch(len){
        case 1:
        assert(dtype==DINT);
        _add<uint8_t>(to,op1,op2);
        return;
        case 2:
        assert(dtype==DINT);
        _add<uint16_t>(to,op1,op2);
        return;
        case 4:
        switch(dtype){
            case DINT: _add<uint32_t>(to,op1,op2);return;
            case DFLOAT: _add<float>(to,op1,op2);return;
            default: assert(0);
        }
        return;
        case 8:
        switch(dtype){
            case DINT: _add<uint64_t>(to,op1,op2);return;
            case DFLOAT: _add<double>(to,op1,op2);return;
            default: assert(0);
        }
        default: assert(0);
        }
    }

    template<typename TOUT,typename TIN>
    void _mulIntWide(void *to,void *op1,void *op2){
        __uint128_t t1,t2;
        t1 = *(TIN*)op1;
        t2 = *(TIN*)op2;
        *(TOUT*)to = (TOUT)(t1 * t2);
    }

    template<typename T>
    void _mulIntHI(void *to,void *op1,void *op2){
        __uint128_t t1,t2;
        t1 = *(T*)op1;
        t2 = *(T*)op2;
        *(T*)to = (T)((t1*t2) >> sizeof(T));
    }

    template<typename T>
    void _mulIntLO(void *to,void *op1,void *op2){
        *(T*)to = *(T*)op1 * *(T*)op2;
    }

    template<typename T>
    void _mulfloat(void *to,void *op1,void *op2){
        *(T*)to = *(T*)op1 * *(T*)op2;
    }
    
    // TODO implement float Qualifier and fix float precision loss
    void mul(void *to,void *op1,void *op2,std::vector<Qualifier>&q){
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        Qualifier mulType;
        if(dtype==DINT)mulType = getMulQ(q);
        switch(len){
        case 1:
        switch(mulType){
            case Q_WIDE:_mulIntWide<uint16_t,uint8_t>(to,op1,op2);return;
            case Q_HI:_mulIntHI<uint8_t>(to,op1,op2);return;
            case Q_LO:_mulIntLO<uint8_t>(to,op1,op2);return;
        }
        return;
        case 2:
        switch(mulType){
            case Q_WIDE:_mulIntWide<uint32_t,uint16_t>(to,op1,op2);return;
            case Q_HI:_mulIntHI<uint16_t>(to,op1,op2);return;
            case Q_LO:_mulIntLO<uint16_t>(to,op1,op2);return;
        }
        return;
        case 4:
        switch(dtype){
            case DINT: 
            switch(mulType){
                case Q_WIDE:_mulIntWide<uint64_t,uint32_t>(to,op1,op2);return;
                case Q_HI:_mulIntHI<uint32_t>(to,op1,op2);return;
                case Q_LO:_mulIntLO<uint32_t>(to,op1,op2);return;
            }
            case DFLOAT: _mulfloat<float>(to,op1,op2);return;
            default: assert(0);
        }
        case 8:
        switch(dtype){
            case DINT: 
            switch(mulType){
                case Q_WIDE:_mulIntWide<__uint128_t,uint64_t>(to,op1,op2);return;
                case Q_HI:_mulIntHI<uint64_t>(to,op1,op2);return;
                case Q_LO:_mulIntLO<uint64_t>(to,op1,op2);return;
            }
            case DFLOAT: _mulfloat<double>(to,op1,op2);return;
            default: assert(0);
        }
        default:assert(0);
        }
    }

    template<typename T>
    void _mov(void *from,void *to){
        *(T*)to = *(T*)from;
    }

    void mov(void *from,void *to,std::vector<Qualifier>&q){
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        switch(len){
        case 1:{
            assert(dtype==DINT);
            _mov<uint8_t>(from,to);
            return;
        }
        case 2:{
            assert(dtype==DINT);
            _mov<uint16_t>(from,to);
            return;
        }
        case 4:{
            if(dtype==DINT)
                _mov<uint32_t>(from,to);
            else if(dtype==DFLOAT)
                _mov<float>(from,to);
            else assert(0);
            return;
        }
        case 8:{
            if(dtype==DINT)
                _mov<uint64_t>(from,to);
            else if(dtype==DFLOAT)
                _mov<double>(from,to);
            else assert(0);
            return;
        }
        default:assert(0);
        }
    }

};
#endif