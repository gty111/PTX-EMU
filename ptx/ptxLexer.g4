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

MINNCTAPERSM: '.minnctapersm' ;

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

GEU: '.geu' ;

RN: '.rn' ;

EQ: '.eq' ;

UNI: '.uni' ;

NE: '.ne' ;

STORE: '.store' ;

LOAD: '.load' ;

A: '.a' ;

B: '.b' ;

D: '.d' ;

MMA: '.mma' ;

ROW: '.row' ;

ALIGNED: '.aligned' ;

M8N8K4: '.m8n8k4' ;

M16N16K16: '.m16n16k16' ;

NEU: '.neu' ;

NC: '.nc' ;

APPROX: '.approx' ;

FTZ: '.ftz' ;

LTU: '.ltu' ;

GTU: '.gtu' ;

LE: '.le' ;

LEU: '.leu' ;

DOTADD: '.add';

DOTOR: '.or';

RZI: '.rzi' ;

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

WMMA: 'wmma' ;

NEG: 'neg' ;

NOT: 'not' ;

SQRT: 'sqrt' ;

COS: 'cos' ;

LG2: 'lg2' ;

EX2: 'ex2' ;

ATOM: 'atom' ;

XOR: 'xor' ;

ABS: 'abs' ;

SIN: 'sin' ;

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

F8: '.f8' ;

F16: '.f16' ;

F32: '.f32' ;

F64: '.f64' ;

S8: '.s8' ;

S16: '.s16' ;

S32: '.s32' ;

S64: '.s64' ;

V2: '.v2' ;

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