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

kernel : VISIBLE? ENTRY ID LeftParen param? RightParen (MAXNTID DIGITS COMMA DIGITS COMMA DIGITS)? compoundStatement 
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
          | F32
          | F64
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
          ;
        
param : qualifier* ID COMMA param
      | qualifier* ID 
      ;   

compoundStatement : LeftBrace statements? RightBrace ;

statements : statement statements
           | statement
           ; 

statement : REG qualifier reg (LESS DIGITS GREATER)? SEMI
          | SHARED ALIGN DIGITS qualifier ID LeftBracket DIGITS RightBracket SEMI
          | LOCAL ALIGN DIGITS qualifier ID LeftBracket DIGITS RightBracket SEMI
          | DOLLOR ID COLON
          | AT reg BRA DOLLOR ID SEMI 
          | PRAGMA STRING SEMI
          | RET SEMI
          | BAR qualifier* DIGITS SEMI
          | BRA qualifier* DOLLOR ID SEMI
          | RCP qualifier* reg COMMA reg SEMI
          | LD qualifier* reg COMMA fetchAddress SEMI
          | MOV qualifier* reg COMMA (reg|ID|DIGITS) SEMI 
          | SETP qualifier* reg COMMA reg COMMA (reg|DIGITS) SEMI
          | (CVTA|CVT) qualifier* reg COMMA reg SEMI
          | (MUL|DIV|SUB|ADD|SHL|SHR|MAX|MIN|AND|OR) qualifier* reg COMMA reg COMMA (reg|DIGITS) SEMI 
          | ST qualifier* fetchAddress COMMA (reg|vector) SEMI
          | (SELP|MAD|FMA) qualifier* reg COMMA (reg|DIGITS) COMMA (reg|DIGITS) COMMA reg SEMI
          ;

reg : PERCENT (ID | ID DOT ID) ;
vector : LeftBrace reg COMMA reg COMMA reg COMMA reg RightBrace ;

fetchAddress : LeftBracket (ID|reg|reg PLUS MINUS? DIGITS) RightBracket ;