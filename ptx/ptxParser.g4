parser grammar ptxParser;

options {
	tokenVocab = ptxLexer;
}

ast : versionDes targetDes addressDes kernels ;

versionDes : VERSION DIGITS DOT DIGITS ;
targetDes : TARGET ID ;
addressDes : ADDRESSSIZE DIGITS ;

kernels : kernel kernels
        | kernel 
        ;

kernel : VISIBLE? ENTRY? ID LeftParen params? RightParen performanceTuning? compoundStatement 
       ;

performanceTuning : MAXNTID DIGITS COMMA DIGITS COMMA DIGITS ;

qualifier : U64
          | U32
          | U16
          | U8
          | PRED
          | B8
          | B16
          | B32
          | B64
          | F8
          | F16
          | F32
          | F64
          | S8
          | S16
          | S32
          | S64
          | V4
          | PARAM
          | GLOBAL
          | LOCAL
          | SHARED
          | GT
          | GE
          | EQ
          | NE
          | LT
          | TO
          | WIDE
          | SYNC
          | LO
          | UNI 
          | RN
          | A
          | B
          | D
          | ROW
          | ALIGNED
          | M8N8K4
          | M16N16K16
          ;

params : param COMMA params
       | param 
       ;

param : PARAM qualifier ID ;

compoundStatement : LeftBrace statements? RightBrace ;

statements : statement statements
           | statement
           ; 

statement : regStatement
          | sharedStatement
          | localStatement
          | dollorStatement
          | atStatement
          | pragmaStatement
          | retStatement
          | barStatement
          | braStatement
          | rcpStatement
          | ldStatement
          | movStatement
          | setpStatement
          | cvtaStatement
          | cvtStatement
          | mulStatement
          | divStatement
          | subStatement
          | addStatement
          | shlStatement
          | shrStatement
          | maxStatement
          | minStatement
          | andStatement
          | orStatement
          | stStatement
          | selpStatement
          | madStatement
          | fmaStatement
          | wmmaStatement
          ;

regStatement : REG qualifier reg (LESS DIGITS GREATER)? SEMI ;
sharedStatement : SHARED ALIGN DIGITS qualifier ID LeftBracket DIGITS RightBracket SEMI ;
localStatement : LOCAL ALIGN DIGITS qualifier ID LeftBracket DIGITS RightBracket SEMI ;
dollorStatement : DOLLOR ID COLON ;
atStatement : AT reg BRA DOLLOR ID SEMI ;
pragmaStatement : PRAGMA STRING SEMI ;
retStatement : RET SEMI ;
barStatement : BAR qualifier+ DIGITS SEMI ;
braStatement : BRA qualifier? DOLLOR ID SEMI ;
rcpStatement : RCP qualifier+ reg COMMA reg SEMI ;
ldStatement : LD qualifier* reg COMMA fetchAddress SEMI ;
movStatement : MOV qualifier reg COMMA (reg|var|imm) SEMI ;
setpStatement : SETP qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
cvtaStatement : CVTA qualifier* reg COMMA reg SEMI ;
cvtStatement : CVT qualifier* reg COMMA reg SEMI ;
mulStatement : MUL qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
divStatement : DIV qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
subStatement : SUB qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
addStatement : ADD qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
shlStatement : SHL qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
shrStatement : SHR qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
maxStatement : MAX qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
minStatement : MIN qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
andStatement : AND qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
orStatement : OR qualifier* reg COMMA reg COMMA (reg|imm) SEMI ;
stStatement : ST qualifier* fetchAddress COMMA (reg|vector) SEMI ;
selpStatement : SELP qualifier reg COMMA (reg|imm) COMMA (reg|imm) COMMA reg SEMI ;
madStatement : MAD qualifier* reg COMMA (reg|imm) COMMA (reg|imm) COMMA reg SEMI ;
fmaStatement : FMA qualifier* reg COMMA (reg|imm) COMMA (reg|imm) COMMA reg SEMI ;
wmmaStatement : WMMA LOAD qualifier* vector COMMA fetchAddress COMMA reg SEMI 
              | WMMA STORE qualifier* fetchAddress COMMA vector COMMA reg SEMI 
              | WMMA MMA qualifier* vector COMMA vector COMMA vector COMMA vector SEMI 
              ;

imm : DIGITS ;

var : ID ;

reg : PERCENT (ID | ID DOT ID) ;

regi : PERCENT (ID | ID DOT ID) ; 

vector : LeftBrace regi RightBrace 
       | LeftBrace regi COMMA regi RightBrace
       | LeftBrace regi COMMA regi COMMA regi COMMA regi RightBrace 
       | LeftBrace regi COMMA regi COMMA regi COMMA regi COMMA regi COMMA regi COMMA regi COMMA regi RightBrace
       ;

fetchAddress : LeftBracket (ID|regi|regi PLUS MINUS? DIGITS) RightBracket ;