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
    DINT
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

    IMM imm;

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
            s->byteNum = Q2bytes(e.paramType);
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
                #ifdef DEBUG
                getchar();
                #endif
                exe_once(e);
                #ifdef DEBUG
                for(auto &e:name2Reg){
                    auto ee = e.second;
                    void *base = ee->addr;
                    for(int i=0;i<ee->elementNum;i++){
                        switch(ee->byteNum){
                        case 4:std::printf("%s%d:%lx\n",ee->name.c_str(),i,*(uint32_t*)(base+i*ee->byteNum));break;
                        case 8:std::printf("%s%d:%lx\n",ee->name.c_str(),i,*(uint64_t*)(base+i*ee->byteNum));break;
                        }
                    }
                }
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
        switch(s.statementType){
        case S_REG:{
            auto ss = (StatementContext::REG*)s.statement;
            if(name2Reg[ss->regName])return; // already alloc
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
            assert(0);
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
            assert(0);
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
            void *op1 = getOperandAddr(ss->shlOp[2],ss->shlQualifier);

            // exe shl
            shl(to,op0,op1,ss->shlQualifier);

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
            void *to = getOperandAddr(ss->stOp[0],ss->stQualifier);
            to = (void*)*(uint64_t*)to;

            // op1
            void *from = getOperandAddr(ss->stOp[1],ss->stQualifier);

            // exe st
            mov(from,to,ss->stQualifier);

            return;
        }
        case S_SELP:{
            assert(0);
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
            void *t = malloc(8);

            mul(t,op0,op1,ss->madQualifier);

            add(to,t,op2,ss->madQualifier);

            free(t);
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

    // helper function

    void setIMM(std::string s,Qualifier q){
        this->imm.type = q;
        switch(q){
        case Q_S64:case Q_U64:case Q_B64:
        this->imm.data.u64 = stol(s,0,0);
        return;
        case Q_S32:case Q_U32:case Q_B32:
        this->imm.data.u32 = stoi(s,0,0);
        return;
        case Q_S16:case Q_U16:case Q_B16:
        this->imm.data.u16 = stoi(s,0,0);
        return;
        case Q_S8:case Q_U8:case Q_B8:
        this->imm.data.u8 = stoi(s,0,0);
        return;
        case Q_F64:
        assert(s.size()==18&&(s[1]=='d'||s[1]=='D'));
        s[1] = 'x';
        *(uint64_t*)&(this->imm.data.f64) = stoull(s,0,0);
        return;
        case Q_F32:
        assert(s.size()==10&&(s[1]=='f'||s[1]=='F'));
        s[1] = 'x';
        *(uint32_t*)&(this->imm.data.f32) = stoi(s,0,0);
        return;
        default:assert(0);
        }
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

    int getBits(std::vector<Qualifier>&q){
        int ret;
        for(auto e:q){
            if(ret=Q2bytes(e))return ret;
        }
        return 0;
    }

    int getBits(Qualifier q){
        std::vector<Qualifier>t;
        t.push_back(q);
        return getBits(t);
    }

    void *getOperandAddr(OperandContext &op,std::vector<Qualifier>&q){
        if(op.opType==O_REG){
            return getRegAddr((OperandContext::REG*)op.operand);
        }else if(op.opType==O_FA){
            return getFaAddr((OperandContext::FA*)op.operand);
        }else if(op.opType==O_IMM){
            setIMM(((OperandContext::IMM*)op.operand)->immVal,
                    getDataType(q));
            return &(this->imm.data);
        }else assert(0);
    }

    void *getRegAddr(OperandContext::REG *regContext){
        if(regContext->regMinorName.size()==1){
            std::printf("INTE: access %s.%s\n",
                regContext->regMajorName.c_str(),regContext->regMinorName.c_str());
            static uint64_t t;
            t = getSpReg(regContext);
            return &t;
        }else{
            std::printf("INTE: access %s%d\n",
                regContext->regMajorName.c_str(),regContext->regIdx);
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
            if(fa->ifMinus){ 
                ret = (void*)((uint64_t)ret + this->imm.data.u64);
            }else{
                ret = (void*)((uint64_t)ret - this->imm.data.u64);
            }
        }
        printf("INTE: FA %p\n",ret);
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
    void _cvt(void *to,void *from,int bitnum){
        switch(bitnum){
        case 8: 
        __cvt<T,uint64_t>(to,from);
        return;
        case 4: 
        __cvt<T,uint32_t>(to,from);
        return;
        case 2:
        __cvt<T,uint16_t>(to,from);
        return;
        case 1:
        __cvt<T,uint8_t>(to,from);
        return;
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
        switch(bitnum[0]){
            case 8: 
            _cvt<uint64_t>(to,from,bitnum[1]);
            return;
            case 4: 
            _cvt<uint32_t>(to,from,bitnum[1]);
            return;
            case 2:
            _cvt<uint16_t>(to,from,bitnum[1]);
            return;
            case 1:
            _cvt<uint8_t>(to,from,bitnum[1]);
            return;
            default: assert(0);
        }
    }

    template<typename T>
    void _add(void *to,void *op1,void *op2){
        *(T*)to = *(T*)op1 + *(T*)op2;
    }

    void add(void *to,void *op1,void *op2,std::vector<Qualifier>&q){
        printf("INTE: add dest:%p op1:%p op2:%p\n",to,op1,op2);
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        switch(len){
        case 1:
        _add<uint8_t>(to,op1,op2);
        return;
        case 2:
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

    void mul(void *to,void *op1,void *op2,std::vector<Qualifier>&q){
        printf("INTE: mul dest:%p op1:%p op2:%p\n",to,op1,op2);
        int len = getBits(q);
        DTYPE dtype = getDType(q);
        Qualifier mulType = getMulQ(q);
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
        std::printf("INTE: mov %p to %p data:%lx\n",from,to,*(T*)from);
    }

    void mov(void *from,void *to,std::vector<Qualifier>&q){
        int len = getBits(q);
        switch(len){
        case 1:
        _mov<uint8_t>(from,to);
        return;
        case 2:
        _mov<uint16_t>(from,to);
        return;
        case 4:
        _mov<uint32_t>(from,to);
        return;
        case 8:
        _mov<uint64_t>(from,to);
        return;
        default:assert(0);
        }
    }

};
#endif