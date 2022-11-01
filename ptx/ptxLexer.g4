lexer grammar ptxLexer;

/* ************ */

VERSION: '.version' ;

SM: 'sm' ;

TARGET: '.target' ;

ADDRESSSIZE: '.address_size' ;

VISIBLE: '.visible' ;

ENTRY: '.entry' ;

PARAM: '.param' ;

MAXNTID: '.maxntid' ;

REG: '.reg' ;

LOCAL: '.local' ;

SHARED: '.shared' ;

GLOBAL: '.global' ;

ALIGN: '.align' ;

GT: '.gt' ;

TO: '.to' ;

WIDE: '.wide' ;

LT: '.lt' ;

SYNC: '.sync' ;

LO: '.lo' ;

GE: '.ge' ;

RN: '.rn' ;

EQ: '.eq' ;

UNI: '.uni' ;

NE: '.ne' ;

/* ************ */

PRAGMA: '.pragma' ;

LD: 'ld' ;

MOV: 'mov' ;

SETP: 'setp' ;

BRA: 'bra' ;

CVTA: 'cvta' ;

MUL: 'mul' ;

ADD: 'add' ;

SHL: 'shl' ;

ST: 'st' ;

BAR: 'bar' ;

MAD: 'mad' ;

DIV: 'div' ;

SUB: 'sub' ;

FMA: 'fma' ;

RET: 'ret' ;

MAX: 'max' ;

SHR: 'shr' ;

AND: 'and' ;

CVT: 'cvt' ;

SELP: 'selp' ;

OR: 'or' ;

MIN: 'min' ;

RCP: 'rcp' ;

/* ************ */
DOUBLEQUOTES: '"' ;

PLUS: '+' ;

MINUS: '-' ;

COLON: ':' ;

DOT: '.' ;

COMMA: ',' ;

UNDERLINE: '_' ;

PERCENT: '%' ;

LESS: '<' ;

GREATER: '>' ;

SEMI: ';' ;

LeftParen: '(';

RightParen: ')';

LeftBrace: '{';

RightBrace: '}';

LeftBracket: '[';

RightBracket: ']';

AT: '@' ;

DOLLOR: '$' ;

/* ************ */

U8: '.u8' ;

U16: '.u16' ;

U32: '.u32' ;

U64: '.u64' ;

PRED: '.pred' ;

B8: '.b8' ;

B16: '.b16' ;

B32: '.b32' ;

B64: '.b64' ;

F32: '.f32' ;

F64: '.f64' ;

S16: '.s16' ;

S32: '.s32' ;

S64: '.s64' ;

V4: '.v4' ;

/* ************ */

STRING: DOUBLEQUOTES (DIGITS|ID) DOUBLEQUOTES ;

DIGITS: MINUS? DIGIT ( NONDIGIT | DIGIT)*;

ID: NONDIGIT ( NONDIGIT | DIGIT)* ;

fragment
NONDIGIT: [a-zA-Z_] ;

fragment
DIGIT: [0-9] ; 

WS: [ \t\r\n]+ -> skip ;