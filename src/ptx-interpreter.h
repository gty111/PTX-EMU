/**
 * @author gtyinstinct
 * ptx interpreter
 * lots of BUG 
*/

#ifndef __PTX_INTERPRETER__
#define __PTX_INTERPRETER__

#include<driver_types.h>
#include<cmath>

#include "ptx-semantic.h"

enum EXE_STATE{
    RUN,
    BAR,
    EXIT
};

enum DTYPE{
    DFLOAT,
    DINT,
    DNONE
};

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

uint64_t SHMEMADDR = 0; // log SHMEMADDR high 32bits

class PtxInterpreter{
  public:
    PtxContext *ptxContext;
    KernelContext *kernelContext;
    void **kernelArgs;
    dim3 gridDim,blockDim;

    std::map<std::string,uint64_t>constName2addr;


    class Symtable{ // integrate param local const
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

    class VEC{
        public:
        std::vector<void *>vec;
    };

    class ThreadContext{
        public:
        std::vector<StatementContext>*statements;
        std::map<std::string,PtxInterpreter::Symtable*>*name2Share;
        std::map<std::string,PtxInterpreter::Symtable*>name2Sym;
        std::map<std::string,PtxInterpreter::Reg*>name2Reg;
        std::map<std::string,int>label2pc;
        dim3 BlockIdx,ThreadIdx,GridDim,BlockDim;
        std::queue<IMM*> imm; // TODO fix memory leak
        std::queue<VEC*> vec; // TODO fix memory leak
        int pc;
        EXE_STATE state;

        void init(dim3 &blockIdx,dim3 &threadIdx,dim3 GridDim,dim3 BlockDim,
            std::vector<StatementContext>&statements,
            std::map<std::string,PtxInterpreter::Symtable*>&name2Share,
            std::map<std::string,PtxInterpreter::Symtable*>&name2Sym,
            std::map<std::string,int>&label2pc){
                
            this->BlockIdx = blockIdx;
            this->ThreadIdx = threadIdx;
            this->GridDim = GridDim;
            this->BlockDim = BlockDim;
            this->name2Share = &name2Share;
            this->name2Sym = name2Sym;
            this->label2pc = label2pc;
            this->statements = &statements;
            this->name2Reg.clear();
            this->pc = -1;
            this->state = RUN;
        }

        void log(){
            std::printf("INTE: BlockIdx(%d,%d,%d) ThreadIdx(%d,%d,%d)\n",
                BlockIdx.x,BlockIdx.y,BlockIdx.z,
                ThreadIdx.x,ThreadIdx.y,ThreadIdx.z);
            std::printf("PC:%d\n",pc);
        }

        void dLog(){
            std::printf("INTE: BlockIdx(%d,%d,%d) ThreadIdx(%d,%d,%d)\n",
                BlockIdx.x,BlockIdx.y,BlockIdx.z,
                ThreadIdx.x,ThreadIdx.y,ThreadIdx.z);
            std::printf("PC:%d\n",pc);
            int toti = 0;
            for(auto &e:name2Reg){
                auto ee = e.second;
                void *base = ee->addr;
                for(int i=0;i<ee->elementNum;i++,toti++){
                    switch(ee->name[0]){
                    case 'r': case 'S':
                    switch(ee->byteNum){
                    case 2:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                        *(uint16_t*)((uint64_t)base+i*ee->byteNum));break;
                    case 4:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                        *(uint32_t*)((uint64_t)base+i*ee->byteNum));break;
                    case 8:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                        *(uint64_t*)((uint64_t)base+i*ee->byteNum));break;
                    default:assert(0);
                    }
                    break;
                    case 'f':
                    switch(ee->byteNum){
                    case 4:std::printf("%5s%-3d:%-16f ",ee->name.c_str(),i,
                        *(float*)((uint64_t)base+i*ee->byteNum));break;
                    case 8:std::printf("%5s%-3d:%-16lf ",ee->name.c_str(),i,
                        *(double*)((uint64_t)base+i*ee->byteNum));break;
                    default: assert(0);
                    }
                    break;
                    case 'p':
                    std::printf("%5s%-3d:%-16d ",ee->name.c_str(),i,
                        *(uint8_t*)((uint64_t)base+i*ee->byteNum));break;
                    default:assert(0);
                    }
                    if((toti+1)%6==0)std::printf("\n");
                }
            }
            std::printf("\n");
        }

        void printReg(std::string &name,int i){
            auto ee = name2Reg[name];
            void *base = name2Reg[name]->addr;
            switch(ee->name[0]){
            case 'r': case 'S':
            switch(ee->byteNum){
            case 2:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                *(uint16_t*)((uint64_t)base+i*ee->byteNum));break;
            case 4:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                *(uint32_t*)((uint64_t)base+i*ee->byteNum));break;
            case 8:std::printf("%5s%-3d:%-16lx ",ee->name.c_str(),i,
                *(uint64_t*)((uint64_t)base+i*ee->byteNum));break;
            default:assert(0);
            }
            break;
            case 'f':
            switch(ee->byteNum){
            case 4:std::printf("%5s%-3d:%-16f ",ee->name.c_str(),i,
                *(float*)((uint64_t)base+i*ee->byteNum));break;
            case 8:std::printf("%5s%-3d:%-16lf ",ee->name.c_str(),i,
                *(double*)((uint64_t)base+i*ee->byteNum));break;
            default: assert(0);
            }
            break;
            case 'p':
            std::printf("%5s%-3d:%-16d ",ee->name.c_str(),i,
                *(uint8_t*)((uint64_t)base+i*ee->byteNum));break;
            default:assert(0);
            }
            std::printf("\n");
        }

        EXE_STATE exe_once(){
            if(state==RUN){
                pc++;
                assert(pc>=0 && pc<(*statements).size());
                auto s = (*statements)[pc];
                #ifdef DEBUGINTE
                std::string cmd,name;
                int i;
                while(1){
                    std::cout << ">>> ";
                    std::cin >> cmd;
                    if(cmd=="s"){
                        _exe_once(s);
                        break;
                    }else {
                        extractREG(cmd,i,name);
                        if(name2Reg[name]){
                            printReg(name,i);
                        }else
                            std::cout << "unrecognized " << cmd << std::endl;
                    }
                }
                
                #endif
                #ifndef DEBUGINTE
                _exe_once(s);
                #endif
            }
            return state;
        }

        void _exe_once(StatementContext &s){
            #ifdef LOGINTE
            std::printf("INTE: %s\n",S2s(s.statementType).c_str());
            #endif
            switch(s.statementType){
            case S_REG:{
                auto ss = (StatementContext::REG*)s.statement;
                assert(ss->regDataType.size()==1);
                PtxInterpreter::Reg *reg = new PtxInterpreter::Reg();
                reg->regType = ss->regDataType.back();
                reg->name = ss->regName;
                reg->elementNum = ss->regNum;
                reg->byteNum = Q2bytes(ss->regDataType.back());
                assert(reg->byteNum&&reg->elementNum);
                reg->addr = calloc(reg->elementNum,reg->byteNum);
                name2Reg[reg->name] = reg;
                return;
            }
            case S_SHARED:{
                auto ss = (StatementContext::SHARED*)s.statement;
                if((*name2Share)[ss->sharedName]){
                    // other thread in same cta alloc before
                    return;
                }
                assert(ss->sharedDataType.size()==1);
                PtxInterpreter::Symtable *share = new PtxInterpreter::Symtable();
                share->byteNum = getBytes(ss->sharedDataType);
                share->elementNum = ss->sharedSize;
                share->name = ss->sharedName;
                share->symType = ss->sharedDataType.back();
                assert(share->byteNum&&share->elementNum);
                share->val = (uint64_t)calloc(share->elementNum,share->byteNum);
                if(SHMEMADDR){
                    assert(share->val >> 32 == SHMEMADDR);
                }else{
                    SHMEMADDR = share->val >> 32; // only save high 32 bit
                }
                share->val = (share->val << 32) >> 32; // only save low 32 bit
                (*name2Share)[share->name] = share;
                return;
            }
            case S_LOCAL:{
                auto ss = (StatementContext::LOCAL*)s.statement;
                assert(ss->localDataType.size()==1);
                PtxInterpreter::Symtable *local = new PtxInterpreter::Symtable();
                local->byteNum = getBytes(ss->localDataType);
                local->elementNum = ss->localSize;
                local->name = ss->localName;
                local->symType = ss->localDataType.back();
                assert(local->byteNum && local->elementNum);
                local->val = (uint64_t)calloc(local->elementNum,local->byteNum);
                name2Sym[local->name] = local;
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
                // temparily do nothing
                return;
            }
            case S_RET:{
                state = EXIT;
                return;
            }
            case S_BAR:{
                state = BAR;
                return;
            }
            case S_BRA:{
                auto ss = (StatementContext::BRA*)s.statement;
                
                //exe bra
                bra(ss->braLabel,ss->braQualifier);
                return;
            }
            case S_RCP:{
                auto ss = (StatementContext::RCP*)s.statement;

                // op0
                void *to = getOperandAddr(ss->rcpOp[0],ss->rcpQualifier);

                // op1
                void *op = getOperandAddr(ss->rcpOp[1],ss->rcpQualifier);

                // exe rcp
                rcp(to,op,ss->rcpQualifier);
                return;
            }
            case S_LD:{
                auto ss = (StatementContext::LD*)s.statement;

                // process op0
                void *to = getOperandAddr(ss->ldOp[0],ss->ldQualifier);

                // process op1
                void *from = getOperandAddr(ss->ldOp[1],ss->ldQualifier);
                if(QvecHasQ(ss->ldQualifier,Q_SHARED)){
                    from = (void *)((uint64_t)from + (SHMEMADDR<<32));
                }

                // exe ld
                if(QvecHasQ(ss->ldQualifier,Q_V2)){
                    uint64_t step = getBytes(ss->ldQualifier);
                    auto vecAddr = this->vec.front()->vec;
                    this->vec.pop();
                    assert(vecAddr.size()==2);
                    for(int i=0;i<2;i++){
                        to = vecAddr[i];
                        mov((void*)((uint64_t)from+i*step),to,ss->ldQualifier);
                    }
                }else if(QvecHasQ(ss->ldQualifier,Q_V4)){
                    uint64_t step = getBytes(ss->ldQualifier);
                    auto vecAddr = this->vec.front()->vec;
                    this->vec.pop();
                    assert(vecAddr.size()==4);
                    for(int i=0;i<4;i++){
                        to = vecAddr[i];
                        mov((void*)((uint64_t)from+i*step),to,ss->ldQualifier);
                    }
                }else{
                    mov(from,to,ss->ldQualifier);
                }
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
                auto ss = (StatementContext::DIV*)s.statement;

                // op0
                void *to = getOperandAddr(ss->divOp[0],ss->divQualifier);

                // op1
                void *op1 = getOperandAddr(ss->divOp[1],ss->divQualifier);

                // op2
                void *op2 = getOperandAddr(ss->divOp[2],ss->divQualifier);

                // exe mul
                div(to,op1,op2,ss->divQualifier);
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
                auto ss = (StatementContext::MAX*)s.statement;

                // op0
                void *to = getOperandAddr(ss->maxOp[0],ss->maxQualifier);

                // op1
                void *op0 = getOperandAddr(ss->maxOp[1],ss->maxQualifier);

                // op2
                void *op1 = getOperandAddr(ss->maxOp[2],ss->maxQualifier);

                // exe max
                max(to,op0,op1,ss->maxQualifier);
                return;
            }
            case S_MIN:{
                auto ss = (StatementContext::MIN*)s.statement;

                // op0
                void *to = getOperandAddr(ss->minOp[0],ss->minQualifier);

                // op1
                void *op0 = getOperandAddr(ss->minOp[1],ss->minQualifier);

                // op2
                void *op1 = getOperandAddr(ss->minOp[2],ss->minQualifier);

                // exe max
                min(to,op0,op1,ss->minQualifier);
                return;
            }
            case S_AND:{
                auto ss = (StatementContext::AND*)s.statement;

                // op0
                void *to = getOperandAddr(ss->andOp[0],ss->andQualifier);

                // op1
                void *op0 = getOperandAddr(ss->andOp[1],ss->andQualifier);

                // op2
                void *op1 = getOperandAddr(ss->andOp[2],ss->andQualifier);

                // exe and 
                And(to,op0,op1,ss->andQualifier);
                return;
            }
            case S_OR:{
                auto ss = (StatementContext::OR*)s.statement;

                // op0
                void *to = getOperandAddr(ss->orOp[0],ss->orQualifier);

                // op1
                void *op0 = getOperandAddr(ss->orOp[1],ss->orQualifier);

                // op2
                void *op1 = getOperandAddr(ss->orOp[2],ss->orQualifier);

                // exe or 
                Or(to,op0,op1,ss->orQualifier);
                return;
            }
            case S_ST:{
                auto ss = (StatementContext::ST*)s.statement;

                // op0
                void *to = getOperandAddr(ss->stOp[0],ss->stQualifier);
                if(QvecHasQ(ss->stQualifier,Q_SHARED)){
                    to = (void*)((uint64_t)to+(SHMEMADDR<<32));
                }

                // op1
                void *from = getOperandAddr(ss->stOp[1],ss->stQualifier);

                // exe st
                if(QvecHasQ(ss->stQualifier,Q_V4)){
                    uint64_t step = getBytes(ss->stQualifier);
                    auto vecAddr = this->vec.front()->vec;
                    this->vec.pop();
                    assert(vecAddr.size()==4);
                    for(int i=0;i<4;i++){
                        from = vecAddr[i];
                        mov(from,(void*)((uint64_t)to+i*step),ss->stQualifier);
                    }
                }else if(QvecHasQ(ss->stQualifier,Q_V2)){
                    uint64_t step = getBytes(ss->stQualifier);
                    auto vecAddr = this->vec.front()->vec;
                    this->vec.pop();
                    assert(vecAddr.size()==2);
                    for(int i=0;i<2;i++){
                        from = vecAddr[i];
                        mov(from,(void*)((uint64_t)to+i*step),ss->stQualifier);
                    }
                }else{
                    mov(from,to,ss->stQualifier);
                }
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

                // exe fma
                fma(to,op0,op1,op2,ss->fmaQualifier);
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
                auto ss = (StatementContext::SQRT*)s.statement;

                // op0
                void *to = getOperandAddr(ss->sqrtOp[0],ss->sqrtQualifier);

                // op1
                void *op = getOperandAddr(ss->sqrtOp[1],ss->sqrtQualifier);

                // exe sqrt
                sqrt(to,op,ss->sqrtQualifier);
                return;
            }
            case S_COS:{
                assert(0);
                return;
            }
            case S_LG2:{
                auto ss = (StatementContext::LG2*)s.statement;

                // op0
                void *to = getOperandAddr(ss->lg2Op[0],ss->lg2Qualifier);

                // op1
                void *op = getOperandAddr(ss->lg2Op[1],ss->lg2Qualifier);

                // exe lg2
                lg2(to,op,ss->lg2Qualifier);
                return;
            }
            case S_EX2:{
                auto ss = (StatementContext::EX2*)s.statement;

                // op0
                void *to = getOperandAddr(ss->ex2Op[0],ss->ex2Qualifier);

                // op1
                void *op = getOperandAddr(ss->ex2Op[1],ss->ex2Qualifier);

                // exe lg2
                ex2(to,op,ss->ex2Qualifier);
                return;
            }
            case S_ATOM:{
                auto ss = (StatementContext::ATOM*)s.statement;

                if(QvecHasQ(ss->atomQualifier,Q_DOTADD)){
                    void *d = getOperandAddr(ss->atomOp[0],ss->atomQualifier);

                    void *a = getOperandAddr(ss->atomOp[1],ss->atomQualifier);

                    void *b = getOperandAddr(ss->atomOp[2],ss->atomQualifier);
                    
                    mov(a,d,ss->atomQualifier); 

                    add(a,a,b,ss->atomQualifier);
                }else assert(0);
        
                return;
            }
            case S_XOR:{
                assert(0);
                return;
            }
            case S_ABS:{
                auto ss = (StatementContext::ABS*)s.statement;

                // op0
                void *to = getOperandAddr(ss->absOp[0],ss->absQualifier);

                // op1
                void *op0 = getOperandAddr(ss->absOp[1],ss->absQualifier);

                // exe abs
                abs(to,op0,ss->absQualifier);

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
            case S_RSQRT:{
                auto ss = (StatementContext::RSQRT*)s.statement;

                // op0
                void *to = getOperandAddr(ss->rsqrtOp[0],ss->rsqrtQualifier);

                // op1
                void *op = getOperandAddr(ss->rsqrtOp[1],ss->rsqrtQualifier);

                // exe rsqrt
                rsqrt(to,op,ss->rsqrtQualifier);

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
            case Q_S8:case Q_U8:case Q_B8:case Q_PRED:
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
                case Q_S8:case Q_B8:case Q_U8:
                case Q_PRED:return DINT;
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

        int getBytes(std::vector<Qualifier>&q){
            int ret;
            for(auto e:q){
                if(ret=Q2bytes(e))return ret;
            }
            return 0;
        }

        int getBytes(Qualifier q){
            return Q2bytes(q);
        }

        void *getOperandAddr(OperandContext &op,std::vector<Qualifier>&q){
            if(op.opType==O_REG){
                return getRegAddr((OperandContext::REG*)op.operand);
            }else if(op.opType==O_FA){
                return getFaAddr((OperandContext::FA*)op.operand,q);
            }else if(op.opType==O_IMM){
                setIMM(((OperandContext::IMM*)op.operand)->immVal,
                        getDataType(q));
                void *t = &(this->imm.front()->data);
                this->imm.pop();
                return t;
            }else if(op.opType==O_VAR){
                auto var = (OperandContext::VAR*)op.operand;
                if((*name2Share)[var->varName]){
                    return &((*name2Share)[((OperandContext::VAR*)op.operand)->varName]->val);
                }else if(name2Sym[var->varName]){
                    return &(name2Sym[var->varName]->val);
                }else assert(0);
            }else if(op.opType==O_VEC){
                PtxInterpreter::VEC *tvec = new PtxInterpreter::VEC();
                auto vecContext = (OperandContext::VEC*)op.operand;
                for(auto e:vecContext->vec){
                    tvec->vec.push_back(getOperandAddr(e,q));
                }
                this->vec.push(tvec);
                return nullptr;
            }
            else assert(0);
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

        int getRegBytes(OperandContext::REG *regContext){
            assert(regContext->regMinorName.size()==0);
            auto reg = name2Reg[regContext->regMajorName];
            return getBytes(reg->regType);
        }

        void *getFaAddr(OperandContext::FA *fa,std::vector<Qualifier>&q){
            void *ret;
            if(fa->reg){
                auto reg = (OperandContext::REG*)fa->reg->operand;
                ret = getRegAddr(reg);
                switch(getRegBytes(reg)){
                case 8: ret = (void*)*(uint64_t*)ret;break;
                case 4: ret = (void*)(uint64_t)*(uint32_t*)ret;break;
                default:assert(0);
                }
            }else{
                #ifdef LOGINTE
                printf("INTE: access %s\n",fa->ID.c_str());
                #endif
                if(name2Sym[fa->ID]){
                    ret = (void*)name2Sym[fa->ID]->val;
                }else if((*name2Share)[fa->ID]){
                    ret = (void*)(*name2Share)[fa->ID]->val;
                }else {
                    assert(0);
                }
            }
            if(fa->offset.size()!=0){
                setIMM(fa->offset,Q_S64);
                IMM *t = this->imm.front();
                this->imm.pop();
                ret = (void*)((uint64_t)ret + t->data.u64);
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
                    return this->ThreadIdx.x;
                }else if(regContext->regMinorName=="y"){
                    return this->ThreadIdx.y;
                }else if(regContext->regMinorName=="z"){
                    return this->ThreadIdx.z;
                }else assert(0);
            }else if(regContext->regMajorName=="ctaid"){
                if(regContext->regMinorName=="x"){
                    return this->BlockIdx.x;
                }else if(regContext->regMinorName=="y"){
                    return this->BlockIdx.y;
                }else if(regContext->regMinorName=="z"){
                    return this->BlockIdx.z;
                }else assert(0);
            }else if(regContext->regMajorName=="ntid"){
                if(regContext->regMinorName=="x"){
                    return this->BlockDim.x;
                }else if(regContext->regMinorName=="y"){
                    return this->BlockDim.y;
                }else if(regContext->regMinorName=="z"){
                    return this->BlockDim.z;
                }else assert(0);
            }else if(regContext->regMajorName=="nctaid"){
                if(regContext->regMinorName=="x"){
                    return this->GridDim.x;
                }else if(regContext->regMinorName=="y"){
                    return this->GridDim.y;
                }else if(regContext->regMinorName=="z"){
                    return this->GridDim.z;
                }else assert(0);
            }else assert(0);
        }

        template<typename T>
        void _abs(void *to,void *op){
            *(T*)to = std::abs(*(T*)op);
        }


        void abs(void *to,void *op,std::vector<Qualifier>&q){
            Qualifier datatype = getDataType(q);
            switch(datatype){
            case Q_S16:_abs<int16_t>(to,op);return;
            case Q_S32:_abs<int32_t>(to,op);return;
            case Q_S64:_abs<int64_t>(to,op);return;
            case Q_F32:_abs<float>(to,op);return;
            case Q_F64:_abs<double>(to,op);return;
            assert(0);
            }
        }

        template<typename T>
        void _ex2(void *to,void *op){
            *(T*)to = std::pow(2,*(T*)op);
        }

        void ex2(void *to,void *op,std::vector<Qualifier>&q){
            Qualifier datatype = getDataType(q);
            assert(datatype==Q_F32);
            _ex2<float>(to,op);
        }

        template<typename T>
        void _lg2(void *to,void *op){
            *(T*)to = std::log2(*(T*)op);
        }

        void lg2(void *to,void *op,std::vector<Qualifier>&q){
            Qualifier datatype = getDataType(q);
            assert(datatype==Q_F32);
            _lg2<float>(to,op);
        }

        template<typename T>
        void _fma(void *to,void *op0,void *op1,void *op2){
            *(T*)to = std::fma(*(T*)op0,*(T*)op1,*(T*)op2);
        }

        void fma(void *to,void *op0,void *op1,void *op2,std::vector<Qualifier>&q){
            int len = getBytes(q);
            assert(getDType(q)==DFLOAT);
            switch(len){
            case 8:{
                _fma<double>(to,op0,op1,op2);
                return;
            }
            case 4:{
                _fma<float>(to,op0,op1,op2);
                return;
            }
            default:assert(0);
            }
        }
        template<typename T>
        void _min(void *to,void *op0,void *op1){
            *(T*)to = std::min( *(T*)op0 , *(T*)op1 ); 
        }

        void min(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
            int len = getBytes(q);
            DTYPE dtype = getDType(q);
            Qualifier datatype = getDataType(q);
            switch(len){
            case 1:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _min<int8_t>(to,op0,op1);
                else 
                    _min<uint8_t>(to,op0,op1);
            } 
            case 2:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _min<int16_t>(to,op0,op1);
                else 
                    _min<uint16_t>(to,op0,op1);
            } 
            case 4:
            switch(dtype){
            case DINT:{
                if(Signed(datatype))
                    _min<int32_t>(to,op0,op1);
                else
                    _min<uint32_t>(to,op0,op1);
                return;
            }
            case DFLOAT:_min<float>(to,op0,op1);return;
            }
            case 8:switch(dtype){
            case DINT:{
                if(Signed(datatype))
                    _min<int64_t>(to,op0,op1);
                else
                    _min<uint64_t>(to,op0,op1);
                return;
            }
            case DFLOAT:_min<double>(to,op0,op1);return;
            }
            default:assert(0);
            }
        }

        template<typename T>
        void _max(void *to,void *op0,void *op1){
            *(T*)to = std::max( *(T*)op0 , *(T*)op1 ); 
        }

        void max(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
            int len = getBytes(q);
            DTYPE dtype = getDType(q);
            Qualifier datatype = getDataType(q);
            switch(len){
            case 1:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _max<int8_t>(to,op0,op1);
                else 
                    _max<uint8_t>(to,op0,op1);
            } 
            case 2:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _max<int16_t>(to,op0,op1);
                else 
                    _max<uint16_t>(to,op0,op1);
            } 
            case 4:
            switch(dtype){
            case DINT:{
                if(Signed(datatype))
                    _max<int32_t>(to,op0,op1);
                else
                    _max<uint32_t>(to,op0,op1);
                return;
            }
            case DFLOAT:_max<float>(to,op0,op1);return;
            }
            case 8:switch(dtype){
            case DINT:{
                if(Signed(datatype))
                    _max<int64_t>(to,op0,op1);
                else
                    _max<uint64_t>(to,op0,op1);
                return;
            }
            case DFLOAT:_max<double>(to,op0,op1);return;
            }
            default:assert(0);
            }
        }

        template<typename T>
        void _rcp(void *to,void *op){
            *(T*)to = 1 / *(T*)op;
        }

        void rcp(void *to,void *op,std::vector<Qualifier>&q){
            int len = getBytes(q);
            DTYPE dtype = getDType(q);
            assert(dtype==DFLOAT);
            switch(len){
            case 8: _rcp<double>(to,op);return;
            case 4: _rcp<float>(to,op);return;
            default: assert(0);
            }
        }

        template<typename T>
        void _or(void *to,void *op0,void *op1){
            *(T*)to = *(T*)op0 | *(T*)op1;
        }

        void Or(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
            int len = getBytes(q);
            DTYPE dtype = getDType(q);
            assert(dtype==DINT);
            switch(len){
            case 1:_or<uint8_t>(to,op0,op1);return;
            case 2:_or<uint16_t>(to,op0,op1);return;
            case 4:_or<uint32_t>(to,op0,op1);return;
            case 8:_or<uint64_t>(to,op0,op1);return;
            default:assert(0);
            }
        }

        template<typename T>
        void _and(void *to,void *op0,void *op1){
            *(T*)to = *(T*)op0 & *(T*)op1;
        }

        void And(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
            int len = getBytes(q);
            DTYPE dtype = getDType(q);
            assert(dtype==DINT);
            switch(len){
            case 1:_and<uint8_t>(to,op0,op1);return;
            case 2:_and<uint16_t>(to,op0,op1);return;
            case 4:_and<uint32_t>(to,op0,op1);return;
            case 8:_and<uint64_t>(to,op0,op1);return;
            default:assert(0);
            }
        }

        void at(void *pred,std::string &label){
            if(*(uint8_t*)pred){
                pc = label2pc[label];
                assert(pc!=0);
            }
        }

        void bra(std::string &braLabel,std::vector<Qualifier>&q){
            pc = label2pc[braLabel];
            assert(pc!=0);
        }

        template<typename T>
        void _sqrt(void *to,void *op){
            *(T*)to = std::sqrt(*(T*)op);
        }

        void sqrt(void *to,void *op,std::vector<Qualifier>&q){
            assert(getDType(q)==DFLOAT);
            int len = getBytes(q);
            switch(len){
            case 4: _sqrt<float>(to,op);return;
            case 8: _sqrt<double>(to,op);return;
            assert(0);
            }
        }

        template<typename T>
        void _rsqrt(void *to,void *op){
            *(T*)to = 1 / std::sqrt(*(T*)op);
        }

        void rsqrt(void *to,void *op,std::vector<Qualifier>&q){
            assert(getDType(q)==DFLOAT);
            int len = getBytes(q);
            switch(len){
            case 4: _rsqrt<float>(to,op);return;
            case 8: _rsqrt<double>(to,op);return;
            assert(0);
            }
        }

        template<typename T>
        void _rem(void *to,void *op0,void *op1){
            *(T*)to = *(T*)op0 % *(T*)op1 ;
        }

        void rem(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
            int len = getBytes(q);
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
            int len = getBytes(q);
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
            case Q_F32:
            _neg<float>(to,op0);
            return;
            case Q_F64:
            _neg<double>(to,op0);
            return;
            default:assert(0);
            }
        }

        Qualifier getCMPOP(std::vector<Qualifier>&q){
            for(auto e:q){
                switch(e){
                case Q_EQ:case Q_NE:case Q_LT:case Q_LE:case Q_GT:
                case Q_GE:case Q_LO:case Q_HI:case Q_LTU:case Q_LEU:
                case Q_GEU:case Q_NEU:case Q_GTU:
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
        void _setp_ne(void *to,void *op0,void *op1){
            *(uint8_t*)to = *(T*)op0 != *(T*)op1;
        }

        template<typename T>
        void _setp_lt(void *to,void *op0,void *op1){
            *(uint8_t*)to = *(T*)op0 < *(T*)op1;
        }

        template<typename T>
        void _setp_le(void *to,void *op0,void *op1){
            *(uint8_t*)to = *(T*)op0 <= *(T*)op1;
        }

        template<typename T>
        void _setp_ge(void *to,void *op0,void *op1){
            *(uint8_t*)to = *(T*)op0 >= *(T*)op1;
        }

        template<typename T>
        void _setp_gt(void *to,void *op0,void *op1){
            *(uint8_t*)to = *(T*)op0 > *(T*)op1;
        }

        void setp(void *to,void *op0,void *op1,std::vector<Qualifier>&q){
            Qualifier cmpOp = getCMPOP(q);
            int len = getBytes(q);
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
                    //assert(dtype==DINT); TODO float comp
                    _setp_eq<uint32_t>(to,op0,op1);
                    return;
                }
                case 8:{
                    //assert(dtype==DINT); TODO double comp
                    _setp_eq<uint64_t>(to,op0,op1);
                    return;
                }
                default:assert(0);
                }
                return;
            }
            case Q_NEU:
            case Q_NE:{ 
                switch(len){
                case 1: {
                    assert(dtype==DINT);
                    _setp_ne<uint8_t>(to,op0,op1);
                    return;
                }
                case 2:{
                    assert(dtype==DINT);
                    _setp_ne<uint16_t>(to,op0,op1);
                    return;
                }
                case 4:{
                    //assert(dtype==DINT); TODO float comp
                    _setp_ne<uint32_t>(to,op0,op1);
                    return;
                }
                case 8:{
                    //assert(dtype==DINT); TODO double comp
                    _setp_ne<uint64_t>(to,op0,op1);
                    return;
                }
                default:assert(0);
                }
                return;
            }
            case Q_LTU:
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
                    if(dtype==DFLOAT){
                        _setp_lt<float>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_lt<int32_t>(to,op0,op1);
                        else
                            _setp_lt<uint32_t>(to,op0,op1);
                    }else assert(0);
                    return;
                }
                case 8:{
                    if(dtype==DFLOAT){
                        _setp_lt<double>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_lt<int64_t>(to,op0,op1);
                        else
                            _setp_lt<uint64_t>(to,op0,op1);
                    }else assert(0);
                }
                default:assert(0);
                }
                return;
            }
            case Q_LEU:
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
                    if(dtype==DFLOAT){
                        _setp_le<float>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_le<int32_t>(to,op0,op1);
                        else
                            _setp_le<uint32_t>(to,op0,op1);
                    }else assert(0);
                    return;
                }
                case 8:{
                    if(dtype==DFLOAT){
                        _setp_le<double>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_le<int64_t>(to,op0,op1);
                        else
                            _setp_le<uint64_t>(to,op0,op1);
                    }else assert(0);
                    return;
                }
                default:assert(0);
                }
                return;
            }
            case Q_GEU:
            case Q_GE:{
                switch(len){
                case 1: {
                    assert(dtype==DINT);
                    if(Signed(datatype))
                        _setp_ge<int8_t>(to,op0,op1);
                    else 
                        _setp_ge<uint8_t>(to,op0,op1);
                    return;
                }
                case 2:{
                    assert(dtype==DINT);
                    if(Signed(datatype))
                        _setp_ge<int16_t>(to,op0,op1);
                    else 
                        _setp_ge<uint16_t>(to,op0,op1);
                    return;
                }
                case 4:{
                    if(dtype==DFLOAT){
                        _setp_ge<float>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_ge<int32_t>(to,op0,op1);
                        else
                            _setp_ge<uint32_t>(to,op0,op1);
                    }else assert(0);
                    return;
                }
                case 8:{
                    if(dtype==DFLOAT){
                        _setp_ge<double>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_ge<int64_t>(to,op0,op1);
                        else
                            _setp_ge<uint64_t>(to,op0,op1);
                    }else assert(0);
                    return;
                }
                default:assert(0);
                }
                return;
            }
            case Q_GTU:
            case Q_GT:{
                switch(len){
                case 1: {
                    assert(dtype==DINT);
                    if(Signed(datatype))
                        _setp_gt<int8_t>(to,op0,op1);
                    else 
                        _setp_gt<uint8_t>(to,op0,op1);
                    return;
                }
                case 2:{
                    assert(dtype==DINT);
                    if(Signed(datatype))
                        _setp_gt<int16_t>(to,op0,op1);
                    else 
                        _setp_gt<uint16_t>(to,op0,op1);
                    return;
                }
                case 4:{
                    if(dtype==DFLOAT){
                        _setp_gt<float>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_gt<int32_t>(to,op0,op1);
                        else
                            _setp_gt<uint32_t>(to,op0,op1);
                    }else assert(0);
                    return;
                }
                case 8:{
                    if(dtype==DFLOAT){
                        _setp_gt<double>(to,op0,op1);
                    }else if(dtype==DINT){
                        if(Signed(datatype))
                            _setp_gt<int64_t>(to,op0,op1);
                        else
                            _setp_gt<uint64_t>(to,op0,op1);
                    }else assert(0);
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
            int len = getBytes(q);
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
            int len = getBytes(q);
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
                if(getBytes(e))bitnum[idx++] = getBytes(e);
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
            int len = getBytes(q);
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
            int len = getBytes(q);
            DTYPE dtype = getDType(q);
            Qualifier datatype = getDataType(q);
            switch(len){
            case 1:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _add<int8_t>(to,op1,op2);
                else
                    _add<uint8_t>(to,op1,op2);
                return;
            }
            case 2:{
                assert(dtype==DINT);
                if(Signed(datatype))
                    _add<int16_t>(to,op1,op2);
                else
                    _add<uint16_t>(to,op1,op2);
                return;
            }
            case 4:
            switch(dtype){
                case DINT: {
                    if(Signed(datatype))
                        _add<int32_t>(to,op1,op2);
                    else
                        _add<uint32_t>(to,op1,op2);
                    return;
                }
                case DFLOAT: _add<float>(to,op1,op2);return;
                default: assert(0);
            }
            return;
            case 8:
            switch(dtype){
                case DINT: {
                    if(Signed(datatype))
                        _add<int64_t>(to,op1,op2);
                    else
                        _add<uint64_t>(to,op1,op2);
                    return;
                }
                case DFLOAT: _add<double>(to,op1,op2);return;
                default: assert(0);
            }
            default: assert(0);
            }
        }

        template<typename T>
        void _div(void *to,void *op0,void *op1){
            *(T*)to = *(T*)op0 / *(T*)op1;
        }

        void div(void *to,void *op1,void *op2,std::vector<Qualifier>&q){
            int len = getBytes(q);
            DTYPE dtype = getDType(q);
            Qualifier datatype = getDataType(q);
            switch(len){
            case 2:{
            assert(dtype==DINT);
            if(Signed(datatype))
                _div<int16_t>(to,op1,op2);
            else
                _div<uint16_t>(to,op1,op2);
            return;
            }
            case 4:
            switch(dtype){
                case DINT: {
                    if(Signed(datatype))
                        _div<int32_t>(to,op1,op2);
                    else
                        _div<uint32_t>(to,op1,op2);
                    return;
                }
                case DFLOAT: _div<float>(to,op1,op2);return;
                default: assert(0);
            }
            return;
            case 8:
            switch(dtype){
                case DINT: {
                    if(Signed(datatype))
                        _div<int64_t>(to,op1,op2);
                    else
                        _div<uint64_t>(to,op1,op2);
                    return;
                }
                case DFLOAT: _div<double>(to,op1,op2);return;
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
            int len = getBytes(q);
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
            int len = getBytes(q);
            switch(len){
            case 1:{
                _mov<uint8_t>(from,to);
                return;
            }
            case 2:{
                _mov<uint16_t>(from,to);
                return;
            }
            case 4:{
                _mov<uint32_t>(from,to);
                return;
            }
            case 8:{
                _mov<uint64_t>(from,to);
                return;
            }
            default:assert(0);
            }
        }
    };

    class CTAContext{
        public:
        ThreadContext *thread=nullptr;
        bool *exitThread=nullptr;
        bool *barThread=nullptr;
        int threadNum,curExeThreadId,exitThreadNum,barThreadNum;
        dim3 blockIdx,GridDim,BlockDim;
        std::map<std::string,PtxInterpreter::Symtable*>name2Share;

        void init(dim3 &GridDim,dim3 &BlockDim,dim3 &blockIdx,
            std::vector<StatementContext>&statements,
            std::map<std::string,PtxInterpreter::Symtable*>&name2Sym,
            std::map<std::string,int>&label2pc){
            
            threadNum = BlockDim.x * BlockDim.y * BlockDim.z;
            curExeThreadId = 0;
            exitThreadNum = 0;
            barThreadNum = 0;

            this->GridDim = GridDim;
            this->BlockDim = BlockDim;
            this->blockIdx = blockIdx;

            // init thread
            assert(threadNum>0 && threadNum<=2048);
            if(!thread)thread = new ThreadContext[threadNum];
            if(!exitThread)exitThread = new bool[threadNum];
            if(!barThread)barThread = new bool[threadNum];
            memset(exitThread,0,sizeof(bool)*threadNum);
            memset(barThread,0,sizeof(bool)*threadNum);
            dim3 threadIdx;
            for(int i=0;i<threadNum;i++){
                threadIdx.z = i / (BlockDim.x*BlockDim.y);
                threadIdx.y = i % (BlockDim.x*BlockDim.y) / (BlockDim.x);
                threadIdx.x = i % (BlockDim.x*BlockDim.y) % (BlockDim.x);
                thread[i].init(blockIdx,threadIdx,GridDim,BlockDim,
                    statements,name2Share,name2Sym,label2pc);
            }
            
        }

        EXE_STATE exe_once(){
            if(exitThreadNum==threadNum)return EXIT;
            if(barThreadNum==threadNum){
                #ifdef LOGINTE
                printf("INTE: bar.sync BlockIdx(%d,%d,%d)\n",
                    blockIdx.x,blockIdx.y,blockIdx.z);
                #endif
                for(int i=0;i<threadNum;i++){
                    thread[i].state = RUN;
                    barThread[i] = 0;
                }
                barThreadNum = 0;
            }
            EXE_STATE state = thread[curExeThreadId].exe_once();
            if(state!=RUN){
                if(state==EXIT&&!exitThread[curExeThreadId]){
                    exitThreadNum ++;
                    exitThread[curExeThreadId] = 1;
                }else if(state==BAR&&!barThread[curExeThreadId]){
                    barThreadNum ++;
                    barThread[curExeThreadId] = 1;
                }
                curExeThreadId ++;
                curExeThreadId %= threadNum;
            }
            return RUN;
        }

        ~CTAContext(){
            if(thread)delete[] thread;
            if(exitThread)delete[] exitThread;
        }
    };

    void launchPtxInterpreter(PtxContext &ptx,std::string &kernel,void **args,
        dim3 &gridDim,dim3 &blockDim){

      // init 
      ptxContext = &ptx;
      kernelArgs = args;
      this->gridDim = gridDim;
      this->blockDim = blockDim;
      SHMEMADDR = 0;
      // find KernelContext
      for(auto &e:ptx.ptxKernels){
        if(e.kernelName==kernel){
            kernelContext = &e;
        }
      }
      funcInterpreter();
    }

    void funcInterpreter(){
        std::map<std::string,PtxInterpreter::Symtable*>name2Sym;
        std::map<std::string,int>label2pc;

        // setup symbol for const
        for(auto e:ptxContext->ptxStatements){
            assert(e.statementType==S_CONST);
            Symtable *s = new Symtable();
            auto st = (StatementContext::CONST*)e.statement;
            assert(st->constDataType.size()==1);
            s->name = st->constName;
            s->symType = st->constDataType.back();
            s->elementNum = st->constSize;
            s->byteNum = Q2bytes(st->constDataType.back());
            s->val = constName2addr[s->name];
            assert(s->val);
            name2Sym[s->name] = s;
        }


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
                label2pc[s->dollorName] = i;
            }
        }
        
        int ctaNum = gridDim.x * gridDim.y * gridDim.z;
        CTAContext cta;
        dim3 blockIdx;
        for(int i=0;i<ctaNum;i++){
            blockIdx.z = i / (gridDim.x*gridDim.y);
            blockIdx.y = i % (gridDim.x*gridDim.y) / (gridDim.x);
            blockIdx.x = i % (gridDim.x*gridDim.y) % (gridDim.x);
            cta.init(gridDim,blockDim,blockIdx,kernelContext->kernelStatements,name2Sym,label2pc);
            while(cta.exe_once()!=EXIT);
        }
        
    }

};
#endif