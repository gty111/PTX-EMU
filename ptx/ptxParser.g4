/**
 * @author gtyinstinct
 * parser : define grammar rules for ptx
*/

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

kernel : VISIBLE? ENTRY? ID LeftParen params? RightParen performanceTunings? compoundStatement 
       ;

performanceTunings : performanceTunings performanceTuning
                   | performanceTuning
                   ;

performanceTuning : MAXNTID DIGITS COMMA DIGITS COMMA DIGITS 
                  | MINNCTAPERSM DIGITS
                  ;

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
          | V2
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
          | HI
          | UNI 
          | RN
          | A
          | B
          | D
          | ROW
          | ALIGNED
          | M8N8K4
          | M16N16K16
          | NEU
          | NC
          | FTZ
          | APPROX
          | LTU 
          | LE
          | GTU
          | LEU
          | DOTADD
          | GEU
          | RZI
          | DOTOR
          ;

params : param COMMA params
       | param 
       ;

param : PARAM (ALIGN DIGITS)? qualifier ID (LeftBracket DIGITS RightBracket)?
      ;

compoundStatement : LeftBrace statements? RightBrace 
                  ;

statements : statement statements
           | statement
           ; 

statement : compoundStatement
          | regStatement
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
          | negStatement
          | notStatement
          | sqrtStatement
          | cosStatement
          | lg2Statement
          | ex2Statement
          | atomStatement
          | xorStatement
          | absStatement
          | sinStatement
          | remStatement
          ;

regStatement : REG qualifier reg (LESS DIGITS GREATER)? SEMI ;
sharedStatement : SHARED ALIGN DIGITS qualifier ID (LeftBracket DIGITS RightBracket)? SEMI ;
localStatement : LOCAL ALIGN DIGITS qualifier ID (LeftBracket DIGITS RightBracket)? SEMI ;
dollorStatement : DOLLOR ID COLON ;
atStatement : AT operand BRA DOLLOR ID SEMI ;
pragmaStatement : PRAGMA STRING SEMI ;
retStatement : RET SEMI ;
barStatement : BAR qualifier* DIGITS SEMI ;
braStatement : BRA qualifier* DOLLOR ID SEMI ;
rcpStatement : RCP qualifier* operandTwo SEMI ;
ldStatement : LD qualifier* operandTwo SEMI ;
movStatement : MOV qualifier operandTwo SEMI ;
setpStatement : SETP qualifier* operandThree SEMI ;
cvtaStatement : CVTA qualifier* operandTwo SEMI ;
cvtStatement : CVT qualifier* operandTwo SEMI ;
mulStatement : MUL qualifier* operandThree SEMI ;
divStatement : DIV qualifier* operandThree SEMI ;
subStatement : SUB qualifier* operandThree SEMI ;
addStatement : ADD qualifier* operandThree SEMI ;
shlStatement : SHL qualifier* operandThree SEMI ;
shrStatement : SHR qualifier* operandThree SEMI ;
maxStatement : MAX qualifier* operandThree SEMI ;
minStatement : MIN qualifier* operandThree SEMI ;
andStatement : AND qualifier* operandThree SEMI ;
orStatement : OR qualifier* operandThree SEMI ;
stStatement : ST qualifier* operandTwo SEMI ;
selpStatement : SELP qualifier* operandFour SEMI ;
madStatement : MAD qualifier* operandFour SEMI ;
fmaStatement : FMA qualifier* operandFour SEMI ;
wmmaStatement : WMMA LOAD qualifier* operandThree SEMI 
              | WMMA STORE qualifier* operandThree SEMI 
              | WMMA MMA qualifier* operandFour SEMI 
              ;
negStatement : NEG qualifier* operandTwo SEMI ;
notStatement : NOT qualifier* operandTwo SEMI ;
sqrtStatement : SQRT qualifier* operandTwo SEMI ;
cosStatement : COS qualifier* operandTwo SEMI ;
lg2Statement : LG2 qualifier* operandTwo SEMI ;
ex2Statement : EX2 qualifier* operandTwo SEMI ; 
atomStatement : ATOM qualifier* (operandThree|operandFour) SEMI ;
xorStatement : XOR qualifier* operandThree SEMI ;
absStatement : ABS qualifier* operandTwo SEMI ;
sinStatement : SIN qualifier* operandTwo SEMI ;
remStatement : REM qualifier* operandThree SEMI ;

operandTwo : operand COMMA operand ;
operandThree : operand COMMA operand COMMA operand;
operandFour : operand COMMA operand COMMA operand COMMA operand;

operand : imm
        | var
        | reg
        | vector
        | fetchAddress
        ;

imm : DIGITS ;

var : ID ;

reg : PERCENT (ID | ID DOT ID) ;

regi : PERCENT (ID | ID DOT ID) ; // reg in vector or fetchAddress

vector : LeftBrace regi RightBrace 
       | LeftBrace regi COMMA regi RightBrace
       | LeftBrace regi COMMA regi COMMA regi COMMA regi RightBrace 
       | LeftBrace regi COMMA regi COMMA regi COMMA regi COMMA regi COMMA regi COMMA regi COMMA regi RightBrace
       ;

fetchAddress : LeftBracket (ID|reg) PLUS? MINUS? DIGITS? RightBracket ;
