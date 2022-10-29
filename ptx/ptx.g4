grammar ptx;

ast : versionDes targetDes addressDes kernels ;

versionDes : VERSION Digits DOT Digits ;
targetDes : TARGET 'sm_' Digits ;
addressDes : ADDRESSSIZE Digits ;

kernels : kernel kernels
        | kernel 
        ;

kernel : VISIBLE? ENTRY Id '(' param? ')' (MAXNTID Digits COMMA Digits COMMA Digits)? compoundStatement 
       ;

qualifier : '.u64'
          | '.u32'
          ;
        
param : PARAM qualifier Id COMMA param
      | PARAM qualifier Id 
      ;   

compoundStatement : '{'  '}' ;

Id : UNDERLINE ( Nondigit | Digit)+ ;

fragment
Nondigit : [a-zA-Z_] ;

Digits : Digit+ ;

fragment
Digit : [0-9] ; 

VERSION : '.version' ;
DOT : '.' ;
TARGET : '.target' ;
ADDRESSSIZE : '.address_size' ;
VISIBLE : '.visible' ;
ENTRY : '.entry' ;
PARAM : '.param' ;
MAXNTID : '.maxntid' ;
COMMA : ',' ;
UNDERLINE : '_' ;

WS : [ \t\r\n]+ -> skip ;