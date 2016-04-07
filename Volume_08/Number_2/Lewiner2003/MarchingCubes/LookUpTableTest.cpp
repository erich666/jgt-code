//------------------------------------------------
// LookUpTable test
//------------------------------------------------
//
// Test of the LookUpTable for the MarchingCubes 33 Algorithm
// Version 0.2 - 14/07/2002
//
// Thomas Lewiner thomas.lewiner@polytechnique.org
// Math Dept, PUC-Rio
//
//________________________________________________



#include <GL/glui.h>
#include <stdio.h>
#include "LookUpTable.h"


//________________________________________________
// Global variables

typedef unsigned char uchar ;
uchar  m_case, m_subcase, m_config ;
char  title[64] ;

//________________________________________________



//________________________________________________
// gets the configuration that generated the case

static const char confcase0[2][8] = {
  /* case  0 (  -1 ):   0 */   { 0, 0, 0, 0, 0, 0, 0, 0 },
  /* case  0 (  -1 ): 255 */   { 1, 1, 1, 1, 1, 1, 1, 1 }
};

static const char confcase1[16][8] = {
  /* case  1 (  0 ):   1 */   { 1, 0, 0, 0, 0, 0, 0, 0 },
  /* case  1 (  1 ):   2 */   { 0, 1, 0, 0, 0, 0, 0, 0 },
  /* case  1 (  2 ):   4 */   { 0, 0, 1, 0, 0, 0, 0, 0 },
  /* case  1 (  3 ):   8 */   { 0, 0, 0, 1, 0, 0, 0, 0 },
  /* case  1 (  4 ):  16 */   { 0, 0, 0, 0, 1, 0, 0, 0 },
  /* case  1 (  5 ):  32 */   { 0, 0, 0, 0, 0, 1, 0, 0 },
  /* case  1 (  6 ):  64 */   { 0, 0, 0, 0, 0, 0, 1, 0 },
  /* case  1 (  7 ): 128 */   { 0, 0, 0, 0, 0, 0, 0, 1 },
  /* case  1 (  8 ): 127 */   { 1, 1, 1, 1, 1, 1, 1, 0 },
  /* case  1 (  9 ): 191 */   { 1, 1, 1, 1, 1, 1, 0, 1 },
  /* case  1 ( 10 ): 223 */   { 1, 1, 1, 1, 1, 0, 1, 1 },
  /* case  1 ( 11 ): 239 */   { 1, 1, 1, 1, 0, 1, 1, 1 },
  /* case  1 ( 12 ): 247 */   { 1, 1, 1, 0, 1, 1, 1, 1 },
  /* case  1 ( 13 ): 251 */   { 1, 1, 0, 1, 1, 1, 1, 1 },
  /* case  1 ( 14 ): 253 */   { 1, 0, 1, 1, 1, 1, 1, 1 },
  /* case  1 ( 15 ): 254 */   { 0, 1, 1, 1, 1, 1, 1, 1 }
};

static const char confcase2[24][8] = {
  /* case  2 (  0 ):   3 */   { 1, 1, 0, 0, 0, 0, 0, 0 },
  /* case  2 (  1 ):   9 */   { 1, 0, 0, 1, 0, 0, 0, 0 },
  /* case  2 (  2 ):  17 */   { 1, 0, 0, 0, 1, 0, 0, 0 },
  /* case  2 (  3 ):   6 */   { 0, 1, 1, 0, 0, 0, 0, 0 },
  /* case  2 (  4 ):  34 */   { 0, 1, 0, 0, 0, 1, 0, 0 },
  /* case  2 (  5 ):  12 */   { 0, 0, 1, 1, 0, 0, 0, 0 },
  /* case  2 (  6 ):  68 */   { 0, 0, 1, 0, 0, 0, 1, 0 },
  /* case  2 (  7 ): 136 */   { 0, 0, 0, 1, 0, 0, 0, 1 },
  /* case  2 (  8 ):  48 */   { 0, 0, 0, 0, 1, 1, 0, 0 },
  /* case  2 (  9 ): 144 */   { 0, 0, 0, 0, 1, 0, 0, 1 },
  /* case  2 ( 10 ):  96 */   { 0, 0, 0, 0, 0, 1, 1, 0 },
  /* case  2 ( 11 ): 192 */   { 0, 0, 0, 0, 0, 0, 1, 1 },
  /* case  2 ( 12 ):  63 */   { 1, 1, 1, 1, 1, 1, 0, 0 },
  /* case  2 ( 13 ): 159 */   { 1, 1, 1, 1, 1, 0, 0, 1 },
  /* case  2 ( 14 ): 111 */   { 1, 1, 1, 1, 0, 1, 1, 0 },
  /* case  2 ( 15 ): 207 */   { 1, 1, 1, 1, 0, 0, 1, 1 },
  /* case  2 ( 16 ): 119 */   { 1, 1, 1, 0, 1, 1, 1, 0 },
  /* case  2 ( 17 ): 187 */   { 1, 1, 0, 1, 1, 1, 0, 1 },
  /* case  2 ( 18 ): 243 */   { 1, 1, 0, 0, 1, 1, 1, 1 },
  /* case  2 ( 19 ): 221 */   { 1, 0, 1, 1, 1, 0, 1, 1 },
  /* case  2 ( 20 ): 249 */   { 1, 0, 0, 1, 1, 1, 1, 1 },
  /* case  2 ( 21 ): 238 */   { 0, 1, 1, 1, 0, 1, 1, 1 },
  /* case  2 ( 22 ): 246 */   { 0, 1, 1, 0, 1, 1, 1, 1 },
  /* case  2 ( 23 ): 252 */   { 0, 0, 1, 1, 1, 1, 1, 1 }
};

static const char confcase3[24][8] = {
  /* case  3 (  0 ):   5 */   { 1, 0, 1, 0, 0, 0, 0, 0 },
  /* case  3 (  1 ):  33 */   { 1, 0, 0, 0, 0, 1, 0, 0 },
  /* case  3 (  2 ): 129 */   { 1, 0, 0, 0, 0, 0, 0, 1 },
  /* case  3 (  3 ):  10 */   { 0, 1, 0, 1, 0, 0, 0, 0 },
  /* case  3 (  4 ):  18 */   { 0, 1, 0, 0, 1, 0, 0, 0 },
  /* case  3 (  5 ):  66 */   { 0, 1, 0, 0, 0, 0, 1, 0 },
  /* case  3 (  6 ):  36 */   { 0, 0, 1, 0, 0, 1, 0, 0 },
  /* case  3 (  7 ): 132 */   { 0, 0, 1, 0, 0, 0, 0, 1 },
  /* case  3 (  8 ):  24 */   { 0, 0, 0, 1, 1, 0, 0, 0 },
  /* case  3 (  9 ):  72 */   { 0, 0, 0, 1, 0, 0, 1, 0 },
  /* case  3 ( 10 ):  80 */   { 0, 0, 0, 0, 1, 0, 1, 0 },
  /* case  3 ( 11 ): 160 */   { 0, 0, 0, 0, 0, 1, 0, 1 },
  /* case  3 ( 12 ):  95 */   { 1, 1, 1, 1, 1, 0, 1, 0 },
  /* case  3 ( 13 ): 175 */   { 1, 1, 1, 1, 0, 1, 0, 1 },
  /* case  3 ( 14 ): 183 */   { 1, 1, 1, 0, 1, 1, 0, 1 },
  /* case  3 ( 15 ): 231 */   { 1, 1, 1, 0, 0, 1, 1, 1 },
  /* case  3 ( 16 ): 123 */   { 1, 1, 0, 1, 1, 1, 1, 0 },
  /* case  3 ( 17 ): 219 */   { 1, 1, 0, 1, 1, 0, 1, 1 },
  /* case  3 ( 18 ): 189 */   { 1, 0, 1, 1, 1, 1, 0, 1 },
  /* case  3 ( 19 ): 237 */   { 1, 0, 1, 1, 0, 1, 1, 1 },
  /* case  3 ( 20 ): 245 */   { 1, 0, 1, 0, 1, 1, 1, 1 },
  /* case  3 ( 21 ): 126 */   { 0, 1, 1, 1, 1, 1, 1, 0 },
  /* case  3 ( 22 ): 222 */   { 0, 1, 1, 1, 1, 0, 1, 1 },
  /* case  3 ( 23 ): 250 */   { 0, 1, 0, 1, 1, 1, 1, 1 }
};

static const char confcase4[8][8] = {
  /* case  4 (  0 ):  65 */   { 1, 0, 0, 0, 0, 0, 1, 0 },
  /* case  4 (  1 ): 130 */   { 0, 1, 0, 0, 0, 0, 0, 1 },
  /* case  4 (  2 ):  20 */   { 0, 0, 1, 0, 1, 0, 0, 0 },
  /* case  4 (  3 ):  40 */   { 0, 0, 0, 1, 0, 1, 0, 0 },
  /* case  4 (  4 ): 215 */   { 1, 1, 1, 0, 1, 0, 1, 1 },
  /* case  4 (  5 ): 235 */   { 1, 1, 0, 1, 0, 1, 1, 1 },
  /* case  4 (  6 ): 125 */   { 1, 0, 1, 1, 1, 1, 1, 0 },
  /* case  4 (  7 ): 190 */   { 0, 1, 1, 1, 1, 1, 0, 1 }
};

static const char confcase5[48][8] = {
  /* case  5 (  0 ):   7 */   { 1, 1, 1, 0, 0, 0, 0, 0 },
  /* case  5 (  1 ):  11 */   { 1, 1, 0, 1, 0, 0, 0, 0 },
  /* case  5 (  2 ):  19 */   { 1, 1, 0, 0, 1, 0, 0, 0 },
  /* case  5 (  3 ):  35 */   { 1, 1, 0, 0, 0, 1, 0, 0 },
  /* case  5 (  4 ):  13 */   { 1, 0, 1, 1, 0, 0, 0, 0 },
  /* case  5 (  5 ):  25 */   { 1, 0, 0, 1, 1, 0, 0, 0 },
  /* case  5 (  6 ): 137 */   { 1, 0, 0, 1, 0, 0, 0, 1 },
  /* case  5 (  7 ):  49 */   { 1, 0, 0, 0, 1, 1, 0, 0 },
  /* case  5 (  8 ): 145 */   { 1, 0, 0, 0, 1, 0, 0, 1 },
  /* case  5 (  9 ):  14 */   { 0, 1, 1, 1, 0, 0, 0, 0 },
  /* case  5 ( 10 ):  38 */   { 0, 1, 1, 0, 0, 1, 0, 0 },
  /* case  5 ( 11 ):  70 */   { 0, 1, 1, 0, 0, 0, 1, 0 },
  /* case  5 ( 12 ):  50 */   { 0, 1, 0, 0, 1, 1, 0, 0 },
  /* case  5 ( 13 ):  98 */   { 0, 1, 0, 0, 0, 1, 1, 0 },
  /* case  5 ( 14 ):  76 */   { 0, 0, 1, 1, 0, 0, 1, 0 },
  /* case  5 ( 15 ): 140 */   { 0, 0, 1, 1, 0, 0, 0, 1 },
  /* case  5 ( 16 ): 100 */   { 0, 0, 1, 0, 0, 1, 1, 0 },
  /* case  5 ( 17 ): 196 */   { 0, 0, 1, 0, 0, 0, 1, 1 },
  /* case  5 ( 18 ): 152 */   { 0, 0, 0, 1, 1, 0, 0, 1 },
  /* case  5 ( 19 ): 200 */   { 0, 0, 0, 1, 0, 0, 1, 1 },
  /* case  5 ( 20 ): 112 */   { 0, 0, 0, 0, 1, 1, 1, 0 },
  /* case  5 ( 21 ): 176 */   { 0, 0, 0, 0, 1, 1, 0, 1 },
  /* case  5 ( 22 ): 208 */   { 0, 0, 0, 0, 1, 0, 1, 1 },
  /* case  5 ( 23 ): 224 */   { 0, 0, 0, 0, 0, 1, 1, 1 },
  /* case  5 ( 24 ):  31 */   { 1, 1, 1, 1, 1, 0, 0, 0 },
  /* case  5 ( 25 ):  47 */   { 1, 1, 1, 1, 0, 1, 0, 0 },
  /* case  5 ( 26 ):  79 */   { 1, 1, 1, 1, 0, 0, 1, 0 },
  /* case  5 ( 27 ): 143 */   { 1, 1, 1, 1, 0, 0, 0, 1 },
  /* case  5 ( 28 ):  55 */   { 1, 1, 1, 0, 1, 1, 0, 0 },
  /* case  5 ( 29 ): 103 */   { 1, 1, 1, 0, 0, 1, 1, 0 },
  /* case  5 ( 30 ):  59 */   { 1, 1, 0, 1, 1, 1, 0, 0 },
  /* case  5 ( 31 ): 155 */   { 1, 1, 0, 1, 1, 0, 0, 1 },
  /* case  5 ( 32 ): 115 */   { 1, 1, 0, 0, 1, 1, 1, 0 },
  /* case  5 ( 33 ): 179 */   { 1, 1, 0, 0, 1, 1, 0, 1 },
  /* case  5 ( 34 ): 157 */   { 1, 0, 1, 1, 1, 0, 0, 1 },
  /* case  5 ( 35 ): 205 */   { 1, 0, 1, 1, 0, 0, 1, 1 },
  /* case  5 ( 36 ): 185 */   { 1, 0, 0, 1, 1, 1, 0, 1 },
  /* case  5 ( 37 ): 217 */   { 1, 0, 0, 1, 1, 0, 1, 1 },
  /* case  5 ( 38 ): 241 */   { 1, 0, 0, 0, 1, 1, 1, 1 },
  /* case  5 ( 39 ): 110 */   { 0, 1, 1, 1, 0, 1, 1, 0 },
  /* case  5 ( 40 ): 206 */   { 0, 1, 1, 1, 0, 0, 1, 1 },
  /* case  5 ( 41 ): 118 */   { 0, 1, 1, 0, 1, 1, 1, 0 },
  /* case  5 ( 42 ): 230 */   { 0, 1, 1, 0, 0, 1, 1, 1 },
  /* case  5 ( 43 ): 242 */   { 0, 1, 0, 0, 1, 1, 1, 1 },
  /* case  5 ( 44 ): 220 */   { 0, 0, 1, 1, 1, 0, 1, 1 },
  /* case  5 ( 45 ): 236 */   { 0, 0, 1, 1, 0, 1, 1, 1 },
  /* case  5 ( 46 ): 244 */   { 0, 0, 1, 0, 1, 1, 1, 1 },
  /* case  5 ( 47 ): 248 */   { 0, 0, 0, 1, 1, 1, 1, 1 }
};

static const char confcase6[48][8] = {
  /* case  6 (  0 ):  67 */   { 1, 1, 0, 0, 0, 0, 1, 0 },
  /* case  6 (  1 ): 131 */   { 1, 1, 0, 0, 0, 0, 0, 1 },
  /* case  6 (  2 ):  21 */   { 1, 0, 1, 0, 1, 0, 0, 0 },
  /* case  6 (  3 ):  69 */   { 1, 0, 1, 0, 0, 0, 1, 0 },
  /* case  6 (  4 ):  41 */   { 1, 0, 0, 1, 0, 1, 0, 0 },
  /* case  6 (  5 ):  73 */   { 1, 0, 0, 1, 0, 0, 1, 0 },
  /* case  6 (  6 ):  81 */   { 1, 0, 0, 0, 1, 0, 1, 0 },
  /* case  6 (  7 ):  97 */   { 1, 0, 0, 0, 0, 1, 1, 0 },
  /* case  6 (  8 ): 193 */   { 1, 0, 0, 0, 0, 0, 1, 1 },
  /* case  6 (  9 ):  22 */   { 0, 1, 1, 0, 1, 0, 0, 0 },
  /* case  6 ( 10 ): 134 */   { 0, 1, 1, 0, 0, 0, 0, 1 },
  /* case  6 ( 11 ):  42 */   { 0, 1, 0, 1, 0, 1, 0, 0 },
  /* case  6 ( 12 ): 138 */   { 0, 1, 0, 1, 0, 0, 0, 1 },
  /* case  6 ( 13 ): 146 */   { 0, 1, 0, 0, 1, 0, 0, 1 },
  /* case  6 ( 14 ): 162 */   { 0, 1, 0, 0, 0, 1, 0, 1 },
  /* case  6 ( 15 ): 194 */   { 0, 1, 0, 0, 0, 0, 1, 1 },
  /* case  6 ( 16 ):  28 */   { 0, 0, 1, 1, 1, 0, 0, 0 },
  /* case  6 ( 17 ):  44 */   { 0, 0, 1, 1, 0, 1, 0, 0 },
  /* case  6 ( 18 ):  52 */   { 0, 0, 1, 0, 1, 1, 0, 0 },
  /* case  6 ( 19 ):  84 */   { 0, 0, 1, 0, 1, 0, 1, 0 },
  /* case  6 ( 20 ): 148 */   { 0, 0, 1, 0, 1, 0, 0, 1 },
  /* case  6 ( 21 ):  56 */   { 0, 0, 0, 1, 1, 1, 0, 0 },
  /* case  6 ( 22 ): 104 */   { 0, 0, 0, 1, 0, 1, 1, 0 },
  /* case  6 ( 23 ): 168 */   { 0, 0, 0, 1, 0, 1, 0, 1 },
  /* case  6 ( 24 ):  87 */   { 1, 1, 1, 0, 1, 0, 1, 0 },
  /* case  6 ( 25 ): 151 */   { 1, 1, 1, 0, 1, 0, 0, 1 },
  /* case  6 ( 26 ): 199 */   { 1, 1, 1, 0, 0, 0, 1, 1 },
  /* case  6 ( 27 ): 107 */   { 1, 1, 0, 1, 0, 1, 1, 0 },
  /* case  6 ( 28 ): 171 */   { 1, 1, 0, 1, 0, 1, 0, 1 },
  /* case  6 ( 29 ): 203 */   { 1, 1, 0, 1, 0, 0, 1, 1 },
  /* case  6 ( 30 ): 211 */   { 1, 1, 0, 0, 1, 0, 1, 1 },
  /* case  6 ( 31 ): 227 */   { 1, 1, 0, 0, 0, 1, 1, 1 },
  /* case  6 ( 32 ):  61 */   { 1, 0, 1, 1, 1, 1, 0, 0 },
  /* case  6 ( 33 ):  93 */   { 1, 0, 1, 1, 1, 0, 1, 0 },
  /* case  6 ( 34 ): 109 */   { 1, 0, 1, 1, 0, 1, 1, 0 },
  /* case  6 ( 35 ): 117 */   { 1, 0, 1, 0, 1, 1, 1, 0 },
  /* case  6 ( 36 ): 213 */   { 1, 0, 1, 0, 1, 0, 1, 1 },
  /* case  6 ( 37 ): 121 */   { 1, 0, 0, 1, 1, 1, 1, 0 },
  /* case  6 ( 38 ): 233 */   { 1, 0, 0, 1, 0, 1, 1, 1 },
  /* case  6 ( 39 ):  62 */   { 0, 1, 1, 1, 1, 1, 0, 0 },
  /* case  6 ( 40 ): 158 */   { 0, 1, 1, 1, 1, 0, 0, 1 },
  /* case  6 ( 41 ): 174 */   { 0, 1, 1, 1, 0, 1, 0, 1 },
  /* case  6 ( 42 ): 182 */   { 0, 1, 1, 0, 1, 1, 0, 1 },
  /* case  6 ( 43 ): 214 */   { 0, 1, 1, 0, 1, 0, 1, 1 },
  /* case  6 ( 44 ): 186 */   { 0, 1, 0, 1, 1, 1, 0, 1 },
  /* case  6 ( 45 ): 234 */   { 0, 1, 0, 1, 0, 1, 1, 1 },
  /* case  6 ( 46 ): 124 */   { 0, 0, 1, 1, 1, 1, 1, 0 },
  /* case  6 ( 47 ): 188 */   { 0, 0, 1, 1, 1, 1, 0, 1 }
};

static const char confcase7[16][8] = {
  /* case  7 (  0 ):  37 */   { 1, 0, 1, 0, 0, 1, 0, 0 },
  /* case  7 (  1 ): 133 */   { 1, 0, 1, 0, 0, 0, 0, 1 },
  /* case  7 (  2 ): 161 */   { 1, 0, 0, 0, 0, 1, 0, 1 },
  /* case  7 (  3 ):  26 */   { 0, 1, 0, 1, 1, 0, 0, 0 },
  /* case  7 (  4 ):  74 */   { 0, 1, 0, 1, 0, 0, 1, 0 },
  /* case  7 (  5 ):  82 */   { 0, 1, 0, 0, 1, 0, 1, 0 },
  /* case  7 (  6 ): 164 */   { 0, 0, 1, 0, 0, 1, 0, 1 },
  /* case  7 (  7 ):  88 */   { 0, 0, 0, 1, 1, 0, 1, 0 },
  /* case  7 (  8 ): 167 */   { 1, 1, 1, 0, 0, 1, 0, 1 },
  /* case  7 (  9 ):  91 */   { 1, 1, 0, 1, 1, 0, 1, 0 },
  /* case  7 ( 10 ): 173 */   { 1, 0, 1, 1, 0, 1, 0, 1 },
  /* case  7 ( 11 ): 181 */   { 1, 0, 1, 0, 1, 1, 0, 1 },
  /* case  7 ( 12 ): 229 */   { 1, 0, 1, 0, 0, 1, 1, 1 },
  /* case  7 ( 13 ):  94 */   { 0, 1, 1, 1, 1, 0, 1, 0 },
  /* case  7 ( 14 ): 122 */   { 0, 1, 0, 1, 1, 1, 1, 0 },
  /* case  7 ( 15 ): 218 */   { 0, 1, 0, 1, 1, 0, 1, 1 }
};

static const char confcase8[6][8] = {
  /* case  8 (  0 ):  15 */   { 1, 1, 1, 1, 0, 0, 0, 0 },
  /* case  8 (  1 ):  51 */   { 1, 1, 0, 0, 1, 1, 0, 0 },
  /* case  8 (  2 ): 153 */   { 1, 0, 0, 1, 1, 0, 0, 1 },
  /* case  8 (  3 ): 102 */   { 0, 1, 1, 0, 0, 1, 1, 0 },
  /* case  8 (  4 ): 204 */   { 0, 0, 1, 1, 0, 0, 1, 1 },
  /* case  8 (  5 ): 240 */   { 0, 0, 0, 0, 1, 1, 1, 1 }
};

static const char confcase9[8][8] = {
  /* case  9 (  0 ):  39 */   { 1, 1, 1, 0, 0, 1, 0, 0 },
  /* case  9 (  1 ):  27 */   { 1, 1, 0, 1, 1, 0, 0, 0 },
  /* case  9 (  2 ): 141 */   { 1, 0, 1, 1, 0, 0, 0, 1 },
  /* case  9 (  3 ): 177 */   { 1, 0, 0, 0, 1, 1, 0, 1 },
  /* case  9 (  4 ):  78 */   { 0, 1, 1, 1, 0, 0, 1, 0 },
  /* case  9 (  5 ): 114 */   { 0, 1, 0, 0, 1, 1, 1, 0 },
  /* case  9 (  6 ): 228 */   { 0, 0, 1, 0, 0, 1, 1, 1 },
  /* case  9 (  7 ): 216 */   { 0, 0, 0, 1, 1, 0, 1, 1 }
};

static const char confcase10[6][8] = {
  /* case 10 (  0 ): 195 */   { 1, 1, 0, 0, 0, 0, 1, 1 },
  /* case 10 (  1 ):  85 */   { 1, 0, 1, 0, 1, 0, 1, 0 },
  /* case 10 (  2 ): 105 */   { 1, 0, 0, 1, 0, 1, 1, 0 },
  /* case 10 (  3 ): 150 */   { 0, 1, 1, 0, 1, 0, 0, 1 },
  /* case 10 (  4 ): 170 */   { 0, 1, 0, 1, 0, 1, 0, 1 },
  /* case 10 (  5 ):  60 */   { 0, 0, 1, 1, 1, 1, 0, 0 }
};

static const char confcase11[12][8] = {
  /* case 11 (  0 ):  23 */   { 1, 1, 1, 0, 1, 0, 0, 0 },
  /* case 11 (  1 ): 139 */   { 1, 1, 0, 1, 0, 0, 0, 1 },
  /* case 11 (  2 ):  99 */   { 1, 1, 0, 0, 0, 1, 1, 0 },
  /* case 11 (  3 ):  77 */   { 1, 0, 1, 1, 0, 0, 1, 0 },
  /* case 11 (  4 ):  57 */   { 1, 0, 0, 1, 1, 1, 0, 0 },
  /* case 11 (  5 ): 209 */   { 1, 0, 0, 0, 1, 0, 1, 1 },
  /* case 11 (  6 ):  46 */   { 0, 1, 1, 1, 0, 1, 0, 0 },
  /* case 11 (  7 ): 198 */   { 0, 1, 1, 0, 0, 0, 1, 1 },
  /* case 11 (  8 ): 178 */   { 0, 1, 0, 0, 1, 1, 0, 1 },
  /* case 11 (  9 ): 156 */   { 0, 0, 1, 1, 1, 0, 0, 1 },
  /* case 11 ( 10 ): 116 */   { 0, 0, 1, 0, 1, 1, 1, 0 },
  /* case 11 ( 11 ): 232 */   { 0, 0, 0, 1, 0, 1, 1, 1 }
};

static const char confcase12[24][8] = {
  /* case 12 (  0 ): 135 */   { 1, 1, 1, 0, 0, 0, 0, 1 },
  /* case 12 (  1 ):  75 */   { 1, 1, 0, 1, 0, 0, 1, 0 },
  /* case 12 (  2 ):  83 */   { 1, 1, 0, 0, 1, 0, 1, 0 },
  /* case 12 (  3 ): 163 */   { 1, 1, 0, 0, 0, 1, 0, 1 },
  /* case 12 (  4 ):  45 */   { 1, 0, 1, 1, 0, 1, 0, 0 },
  /* case 12 (  5 ):  53 */   { 1, 0, 1, 0, 1, 1, 0, 0 },
  /* case 12 (  6 ): 149 */   { 1, 0, 1, 0, 1, 0, 0, 1 },
  /* case 12 (  7 ): 101 */   { 1, 0, 1, 0, 0, 1, 1, 0 },
  /* case 12 (  8 ): 197 */   { 1, 0, 1, 0, 0, 0, 1, 1 },
  /* case 12 (  9 ):  89 */   { 1, 0, 0, 1, 1, 0, 1, 0 },
  /* case 12 ( 10 ): 169 */   { 1, 0, 0, 1, 0, 1, 0, 1 },
  /* case 12 ( 11 ): 225 */   { 1, 0, 0, 0, 0, 1, 1, 1 },
  /* case 12 ( 12 ):  30 */   { 0, 1, 1, 1, 1, 0, 0, 0 },
  /* case 12 ( 13 ):  86 */   { 0, 1, 1, 0, 1, 0, 1, 0 },
  /* case 12 ( 14 ): 166 */   { 0, 1, 1, 0, 0, 1, 0, 1 },
  /* case 12 ( 15 ):  58 */   { 0, 1, 0, 1, 1, 1, 0, 0 },
  /* case 12 ( 16 ): 154 */   { 0, 1, 0, 1, 1, 0, 0, 1 },
  /* case 12 ( 17 ): 106 */   { 0, 1, 0, 1, 0, 1, 1, 0 },
  /* case 12 ( 18 ): 202 */   { 0, 1, 0, 1, 0, 0, 1, 1 },
  /* case 12 ( 19 ): 210 */   { 0, 1, 0, 0, 1, 0, 1, 1 },
  /* case 12 ( 20 ):  92 */   { 0, 0, 1, 1, 1, 0, 1, 0 },
  /* case 12 ( 21 ): 172 */   { 0, 0, 1, 1, 0, 1, 0, 1 },
  /* case 12 ( 22 ): 180 */   { 0, 0, 1, 0, 1, 1, 0, 1 },
  /* case 12 ( 23 ): 120 */   { 0, 0, 0, 1, 1, 1, 1, 0 }
};

static const char confcase13[2][8] = {
  /* case 13 (  0 ): 165 */   { 1, 0, 1, 0, 0, 1, 0, 1 },
  /* case 13 (  1 ):  90 */   { 0, 1, 0, 1, 1, 0, 1, 0 }
};

static const char confcase14[12][8] = {
  /* case 14 (  0 ):  71 */   { 1, 1, 1, 0, 0, 0, 1, 0 },
  /* case 14 (  1 ):  43 */   { 1, 1, 0, 1, 0, 1, 0, 0 },
  /* case 14 (  2 ): 147 */   { 1, 1, 0, 0, 1, 0, 0, 1 },
  /* case 14 (  3 ):  29 */   { 1, 0, 1, 1, 1, 0, 0, 0 },
  /* case 14 (  4 ): 201 */   { 1, 0, 0, 1, 0, 0, 1, 1 },
  /* case 14 (  5 ): 113 */   { 1, 0, 0, 0, 1, 1, 1, 0 },
  /* case 14 (  6 ): 142 */   { 0, 1, 1, 1, 0, 0, 0, 1 },
  /* case 14 (  7 ):  54 */   { 0, 1, 1, 0, 1, 1, 0, 0 },
  /* case 14 (  8 ): 226 */   { 0, 1, 0, 0, 0, 1, 1, 1 },
  /* case 14 (  9 ): 108 */   { 0, 0, 1, 1, 0, 1, 1, 0 },
  /* case 14 ( 10 ): 212 */   { 0, 0, 1, 0, 1, 0, 1, 1 },
  /* case 14 ( 11 ): 184 */   { 0, 0, 0, 1, 1, 1, 0, 1 }
};
//________________________________________________

static const char confsizes[15] = {
  2, 16, 24, 24, 8, 48, 48, 16, 6, 8, 6, 12, 24, 2, 12
};

static const char subcasesizes[15] = {
  1, 1, 1, 2, 2, 1, 3, 9, 1, 1, 5, 1, 5, 50, 1
};

char* get_confcase( int c, int s )
{
  switch( c )
  {
  case  0: return (char*) confcase0  [s] ;
  case  1: return (char*) confcase1  [s] ;
  case  2: return (char*) confcase2  [s] ;
  case  3: return (char*) confcase3  [s] ;
  case  4: return (char*) confcase4  [s] ;
  case  5: return (char*) confcase5  [s] ;
  case  6: return (char*) confcase6  [s] ;
  case  7: return (char*) confcase7  [s] ;
  case  8: return (char*) confcase8  [s] ;
  case  9: return (char*) confcase9  [s] ;
  case 10: return (char*) confcase10 [s] ;
  case 11: return (char*) confcase11 [s] ;
  case 12: return (char*) confcase12 [s] ;
  case 13: return (char*) confcase13 [s] ;
  case 14: return (char*) confcase14 [s] ;
  default: return (char*)NULL ;
  }
}
//________________________________________________




//________________________________________________
// helper functions declaration

void draw_cube() ;
void draw_triangle( const char* t, char n ) ;
void mark_vertex( char* conf ) ;
void mark_face( char f ) ;
//________________________________________________




//_____________________________________________________________________________
// glBuildList
int   gllist ;
//-----------------------------------------------------------------------------
void glBuildList()
//-----------------------------------------------------------------------------
{
  if( m_case > 14 ) m_case =  0 ;
  if( m_subcase > subcasesizes[m_case]-1 ) m_subcase = 0 ;
  if( m_config  > confsizes   [m_case]-1 ) m_config  = 0 ;

  if(glIsList(gllist)==GL_TRUE)
    glDeleteLists( gllist, 1 ) ;

  if( ( gllist = glGenLists(1) ) == 0)
    return;

  sprintf( title, "Case %d (%d), config %d", m_case, m_subcase, m_config ) ;
  glutSetWindowTitle( title ) ;

  glNewList(gllist,GL_COMPILE);
  {
    mark_vertex( get_confcase( m_case, m_config) ) ;

    switch( m_case )
    {
    case  0 :
      break ;

    case  1 :
      draw_triangle( tiling1[m_config], 1 ) ;
      break ;

    case  2 :
      draw_triangle( tiling2[m_config], 2 ) ;
      break ;

    case  3 :
      switch( m_subcase )
      {
      case 0 :
        if( test3[m_config] ) mark_face( test3[m_config]) ;
        draw_triangle( tiling3_2[m_config], 4 ) ; break ;
      case 1 :
        if( test3[m_config] < 0 ) mark_face(-test3[m_config]) ;
        draw_triangle( tiling3_1[m_config], 2 ) ; break ;
      };
      break ;

    case  4 :
      switch( m_subcase )
      {
      case 0 :
        if( test4[m_config] < 0 ) mark_face(-test4[m_config]) ;
        draw_triangle( tiling4_1[m_config], 2 ) ; break ;
      case 1 :
        if( test4[m_config] > 0 ) mark_face( test4[m_config]) ;
        draw_triangle( tiling4_2[m_config], 6 ) ; break ;
      };
      break ;

    case  5 :
      draw_triangle( tiling5[m_config], 3 ) ;
      break ;

    case  6 :
      switch( m_subcase )
      {
      case 0 :
        if( test6[m_config][0] > 0 ) mark_face( test6[m_config][0]) ;
        draw_triangle( tiling6_2[m_config], 5 ) ; break ;
      case 1 :
        if( test6[m_config][0] < 0 ) mark_face(-test6[m_config][0]) ;
        if( test6[m_config][1] < 0 ) mark_face(-test6[m_config][1]) ;
        draw_triangle( tiling6_1_1[m_config], 3 ) ; break ;
      case 2 :
        if( test6[m_config][0] < 0 ) mark_face(-test6[m_config][0]) ;
        if( test6[m_config][1] > 0 ) mark_face( test6[m_config][1]) ;
        draw_triangle( tiling6_1_2[m_config], 9 ) ; break ;
      };
      break ;

    case  7 :
      switch( m_subcase )
      {
      case 0 :
        if( test7[m_config][0] > 0 ) mark_face( test7[m_config][0]) ;
        if( test7[m_config][1] > 0 ) mark_face( test7[m_config][1]) ;
        if( test7[m_config][2] > 0 ) mark_face( test7[m_config][2]) ;
        if( test7[m_config][3] > 0 ) mark_face( test7[m_config][3]) ;
        draw_triangle( tiling7_4_1[m_config], 5 ) ; break ;
      case 1 :
        if( test7[m_config][0] > 0 ) mark_face( test7[m_config][0]) ;
        if( test7[m_config][1] > 0 ) mark_face( test7[m_config][1]) ;
        if( test7[m_config][2] > 0 ) mark_face( test7[m_config][2]) ;
        if( test7[m_config][3] < 0 ) mark_face(-test7[m_config][3]) ;
        draw_triangle( tiling7_4_2[m_config], 9 ) ; break ;
      case 2 :
        if( test7[m_config][0] > 0 ) mark_face( test7[m_config][0]) ;
        if( test7[m_config][1] > 0 ) mark_face( test7[m_config][1]) ;
        if( test7[m_config][2] < 0 ) mark_face(-test7[m_config][2]) ;
        draw_triangle( tiling7_3[m_config][0], 9 ) ; break ;
      case 3 :
        if( test7[m_config][0] > 0 ) mark_face( test7[m_config][0]) ;
        if( test7[m_config][1] < 0 ) mark_face(-test7[m_config][1]) ;
        if( test7[m_config][2] > 0 ) mark_face( test7[m_config][2]) ;
        draw_triangle( tiling7_3[m_config][1], 9 ) ; break ;
      case 4 :
        if( test7[m_config][0] < 0 ) mark_face(-test7[m_config][0]) ;
        if( test7[m_config][1] > 0 ) mark_face( test7[m_config][1]) ;
        if( test7[m_config][2] > 0 ) mark_face( test7[m_config][2]) ;
        draw_triangle( tiling7_3[m_config][2], 9 ) ; break ;
      case 5 :
        if( test7[m_config][0] > 0 ) mark_face( test7[m_config][0]) ;
        if( test7[m_config][1] < 0 ) mark_face(-test7[m_config][1]) ;
        if( test7[m_config][2] < 0 ) mark_face(-test7[m_config][2]) ;
        draw_triangle( tiling7_2[m_config][0], 5 ) ; break ;
      case 6 :
        if( test7[m_config][0] < 0 ) mark_face(-test7[m_config][0]) ;
        if( test7[m_config][1] > 0 ) mark_face( test7[m_config][1]) ;
        if( test7[m_config][2] < 0 ) mark_face(-test7[m_config][2]) ;
        draw_triangle( tiling7_2[m_config][1], 5 ) ; break ;
      case 7 :
        if( test7[m_config][0] < 0 ) mark_face(-test7[m_config][0]) ;
        if( test7[m_config][1] < 0 ) mark_face(-test7[m_config][1]) ;
        if( test7[m_config][2] > 0 ) mark_face( test7[m_config][2]) ;
        draw_triangle( tiling7_2[m_config][2], 5 ) ; break ;
      case 8 :
        if( test7[m_config][0] < 0 ) mark_face(-test7[m_config][0]) ;
        if( test7[m_config][1] < 0 ) mark_face(-test7[m_config][1]) ;
        if( test7[m_config][2] < 0 ) mark_face(-test7[m_config][2]) ;
        draw_triangle( tiling7_1[m_config], 3 ) ; break ;
      };
      break ;

    case  8 :
      draw_triangle( tiling8[m_config], 2 ) ;
      break ;

    case  9 :
      draw_triangle( tiling9[m_config], 4 ) ;
      break ;

    case 10 :
      switch( m_subcase )
      {
      case 0 :
        if( test10[m_config][0] > 0 ) mark_face( test10[m_config][0]) ;
        if( test10[m_config][1] < 0 ) mark_face(-test10[m_config][1]) ;
        draw_triangle( tiling10_2[m_config], 8 ) ; break ;
      case 1 :
        if( test10[m_config][0] < 0 ) mark_face(-test10[m_config][0]) ;
        if( test10[m_config][1] > 0 ) mark_face( test10[m_config][1]) ;
        draw_triangle( tiling10_2_[m_config], 8 ) ; break ;
      case 2 :
        if( test10[m_config][0] < 0 ) mark_face(-test10[m_config][0]) ;
        if( test10[m_config][1] < 0 ) mark_face(-test10[m_config][1]) ;
        if( test10[m_config][2] < 0 ) mark_face(-test10[m_config][2]) ;
        draw_triangle( tiling10_1_1[m_config], 4 ) ; break ;
      case 3 :
        if( test10[m_config][0] > 0 ) mark_face( test10[m_config][0]) ;
        if( test10[m_config][1] > 0 ) mark_face( test10[m_config][1]) ;
        draw_triangle( tiling10_1_1_[m_config], 4 ) ; break ;
      case 4 :
        if( test10[m_config][0] < 0 ) mark_face(-test10[m_config][0]) ;
        if( test10[m_config][1] < 0 ) mark_face(-test10[m_config][1]) ;
        if( test10[m_config][2] > 0 ) mark_face( test10[m_config][2]) ;
        draw_triangle( tiling10_1_2[m_config], 8 ) ; break ;
      };
      break ;

    case 11 :
      draw_triangle( tiling11[m_config], 4 ) ;
      break ;

    case 12 :
      switch( m_subcase )
      {
      case 0 :
        if( test12[m_config][0] > 0 ) mark_face( test12[m_config][0]) ;
        if( test12[m_config][1] < 0 ) mark_face(-test12[m_config][1]) ;
        draw_triangle( tiling12_2[m_config], 8 ) ; break ;
      case 1 :
        if( test12[m_config][0] < 0 ) mark_face(-test12[m_config][0]) ;
        if( test12[m_config][1] > 0 ) mark_face( test12[m_config][1]) ;
        draw_triangle( tiling12_2_[m_config], 8 ) ; break ;
      case 2 :
        if( test12[m_config][0] < 0 ) mark_face(-test12[m_config][0]) ;
        if( test12[m_config][1] < 0 ) mark_face(-test12[m_config][1]) ;
        if( test12[m_config][2] < 0 ) mark_face(-test12[m_config][2]) ;
        draw_triangle( tiling12_1_1[m_config], 4 ) ; break ;
      case 3 :
        if( test12[m_config][0] > 0 ) mark_face( test12[m_config][0]) ;
        if( test12[m_config][1] > 0 ) mark_face( test12[m_config][1]) ;
        draw_triangle( tiling12_1_1_[m_config], 4 ) ; break ;
      case 4 :
        if( test12[m_config][0] < 0 ) mark_face(-test12[m_config][0]) ;
        if( test12[m_config][1] < 0 ) mark_face(-test12[m_config][1]) ;
        if( test12[m_config][2] > 0 ) mark_face( test12[m_config][2]) ;
        draw_triangle( tiling12_1_2[m_config], 8 ) ; break ;
      };
      break ;

    case 13 :
      switch( m_subcase )
      {
      case 0 : /* 13.1 */
        draw_triangle( tiling13_1[m_config], 4 ) ; break ;

      case 1 :/* 13.2 */
        mark_face( test13[m_config][0]) ;
        draw_triangle( tiling13_2[m_config][0], 6 ) ; break ;
      case 2 :/* 13.2 */
        mark_face( test13[m_config][1]) ;
        draw_triangle( tiling13_2[m_config][1], 6 ) ; break ;
      case 3 :/* 13.2 */
        mark_face( test13[m_config][2]) ;
        draw_triangle( tiling13_2[m_config][2], 6 ) ; break ;
      case 4 :/* 13.2 */
        mark_face( test13[m_config][3]) ;
        draw_triangle( tiling13_2[m_config][3], 6 ) ; break ;
      case 5 :/* 13.2 */
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_2[m_config][4], 6 ) ; break ;
      case 6 :/* 13.2 */
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_2[m_config][5], 6 ) ; break ;

      case 7 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        draw_triangle( tiling13_3[m_config][0], 10 ) ; break ;
      case 8 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][3]) ;
        draw_triangle( tiling13_3[m_config][1], 10 ) ; break ;
      case 9 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3[m_config][2], 10 ) ; break ;
      case 10 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3[m_config][3], 10 ) ; break ;
      case 11 :/* 13.3 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        draw_triangle( tiling13_3[m_config][4], 10 ) ; break ;
      case 12 :/* 13.3 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3[m_config][5], 10 ) ; break ;
      case 13 :/* 13.3 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3[m_config][6], 10 ) ; break ;
      case 14 :/* 13.3 */
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        draw_triangle( tiling13_3[m_config][7], 10 ) ; break ;
      case 15 :/* 13.3 */
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3[m_config][8], 10 ) ; break ;
      case 16 :/* 13.3 */
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3[m_config][9], 10 ) ; break ;
      case 17 :/* 13.3 */
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3[m_config][10], 10 ) ; break ;
      case 18 :/* 13.3 */
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3[m_config][11], 10 ) ; break ;

      case 19 :/* 13.4 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_4[m_config][0], 12 ) ; break ;
      case 20 :/* 13.4 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_4[m_config][1], 12 ) ; break ;
      case 21 :/* 13.4 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_4[m_config][2], 12 ) ; break ;
      case 22 :/* 13.4 */
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_4[m_config][3], 12 ) ; break ;

      case 23 :/* 13.5.1 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_5_1[m_config][0], 6 ) ; break ;
      case 24 :/* 13.5.1 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_5_1[m_config][1], 6 ) ; break ;
      case 25 :/* 13.5.1 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_5_1[m_config][2], 6 ) ; break ;
      case 26 :/* 13.5.1 */
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_5_1[m_config][3], 6 ) ; break ;

      case 27 :/* 13.5.2 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][6]) ;
        draw_triangle( tiling13_5_2[m_config][0], 10 ) ; break ;
      case 28 :/* 13.5.2 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        mark_face( test13[m_config][6]) ;
        draw_triangle( tiling13_5_2[m_config][1], 10 ) ; break ;
      case 29 :/* 13.5.2 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][5]) ;
        mark_face( test13[m_config][6]) ;
        draw_triangle( tiling13_5_2[m_config][2], 10 ) ; break ;
      case 30 :/* 13.5.2 */
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][6]) ;
        draw_triangle( tiling13_5_2[m_config][3], 10 ) ; break ;

      case 31 :/* 13.3 */
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][0], 10 ) ; break ;
      case 32 :/* 13.3 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][1], 10 ) ; break ;
      case 33 :/* 13.3 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][2], 10 ) ; break ;
      case 34 :/* 13.3 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3_[m_config][3], 10 ) ; break ;
      case 35 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][4], 10 ) ; break ;
      case 36 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][5], 10 ) ; break ;
      case 37 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3_[m_config][6], 10 ) ; break ;
      case 38 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][7], 10 ) ; break ;
      case 39 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][8], 10 ) ; break ;
      case 40 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3_[m_config][9], 10 ) ; break ;
      case 41 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_3_[m_config][10], 10 ) ; break ;
      case 42 :/* 13.3 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_3_[m_config][11], 10 ) ; break ;


      case 43 :/* 13.2 */
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_2_[m_config][0], 6 ) ; break ;
      case 44 :/* 13.2 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_2_[m_config][1], 6 ) ; break ;
      case 45 :/* 13.2 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_2_[m_config][2], 6 ) ; break ;
      case 46 :/* 13.2 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_2_[m_config][3], 6 ) ; break ;
      case 47 :/* 13.2 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_2_[m_config][4], 6 ) ; break ;
      case 48 :/* 13.2 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        draw_triangle( tiling13_2_[m_config][5], 6 ) ; break ;
      case 49 :/* 13.1 */
        mark_face( test13[m_config][0]) ;
        mark_face( test13[m_config][1]) ;
        mark_face( test13[m_config][2]) ;
        mark_face( test13[m_config][3]) ;
        mark_face( test13[m_config][4]) ;
        mark_face( test13[m_config][5]) ;
        draw_triangle( tiling13_1_[m_config], 4 ) ; break ;
      }
      break ;

    case 14 :
      draw_triangle( tiling14[m_config], 4 ) ;
      break ;
    };
  }
  glEndList();
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Drawing Elements
void draw_triangle( const char* t, char n )
//-----------------------------------------------------------------------------
{
  int i ;
  glColor4f( 1.0f, 1.0f, 0.0f, 0.4f ) ;
  glBegin( GL_TRIANGLES ) ;
  {
    for(i = 0 ; i < 3*n ; i++ )
    {
      switch( t[i] )
      {
      case  0 : glVertex3f( 0.5f, 0.0f, 0.0f ) ; break ;
      case  1 : glVertex3f( 1.0f, 0.5f, 0.0f ) ; break ;
      case  2 : glVertex3f( 0.5f, 1.0f, 0.0f ) ; break ;
      case  3 : glVertex3f( 0.0f, 0.5f, 0.0f ) ; break ;
      case  4 : glVertex3f( 0.5f, 0.0f, 1.0f ) ; break ;
      case  5 : glVertex3f( 1.0f, 0.5f, 1.0f ) ; break ;
      case  6 : glVertex3f( 0.5f, 1.0f, 1.0f ) ; break ;
      case  7 : glVertex3f( 0.0f, 0.5f, 1.0f ) ; break ;
      case  8 : glVertex3f( 0.0f, 0.0f, 0.5f ) ; break ;
      case  9 : glVertex3f( 1.0f, 0.0f, 0.5f ) ; break ;
      case 10 : glVertex3f( 1.0f, 1.0f, 0.5f ) ; break ;
      case 11 : glVertex3f( 0.0f, 1.0f, 0.5f ) ; break ;
      case 12 : glVertex3f( 0.5f, 0.5f, 0.5f ) ; break ;
      };
    }
  }
  glEnd() ;


  glColor4f( 1.0f, 0.0f, 0.0f, 0.4f ) ;
  glBegin( GL_LINE_LOOP ) ;
  {
    for( i = 0 ; i < 3*n ; i++ )
    {
      switch( t[i] )
      {
      case  0 : glVertex3f( 0.5f, 0.0f, 0.0f ) ; break ;
      case  1 : glVertex3f( 1.0f, 0.5f, 0.0f ) ; break ;
      case  2 : glVertex3f( 0.5f, 1.0f, 0.0f ) ; break ;
      case  3 : glVertex3f( 0.0f, 0.5f, 0.0f ) ; break ;
      case  4 : glVertex3f( 0.5f, 0.0f, 1.0f ) ; break ;
      case  5 : glVertex3f( 1.0f, 0.5f, 1.0f ) ; break ;
      case  6 : glVertex3f( 0.5f, 1.0f, 1.0f ) ; break ;
      case  7 : glVertex3f( 0.0f, 0.5f, 1.0f ) ; break ;
      case  8 : glVertex3f( 0.0f, 0.0f, 0.5f ) ; break ;
      case  9 : glVertex3f( 1.0f, 0.0f, 0.5f ) ; break ;
      case 10 : glVertex3f( 1.0f, 1.0f, 0.5f ) ; break ;
      case 11 : glVertex3f( 0.0f, 1.0f, 0.5f ) ; break ;
      case 12 : glVertex3f( 0.5f, 0.5f, 0.5f ) ; break ;
      };
      if( i%3 ==2 ) { glEnd() ; glBegin( GL_LINE_LOOP ) ; }
    }
  }
  glEnd() ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// draw_cube
void draw_cube()
//-----------------------------------------------------------------------------
{
  ::glBegin(GL_LINES);
  {
    ::glColor3f(1,0,0);
    ::glVertex3f( 0 ,0,0); // e0
    ::glVertex3f(1.1f,0,0); // e1

    ::glColor3f(1,0,0);
    ::glVertex3f(0,1,0); // e2
    ::glVertex3f(1,1,0); // e3

    ::glColor3f(1,0,0);
    ::glVertex3f(0,1,1); // e6
    ::glVertex3f(1,1,1); // e7

    ::glColor3f(1,0,0);
    ::glVertex3f(0,0,1); // e4
    ::glVertex3f(1,0,1); // e5

/*---------------------------------------------------------------*/

    ::glColor3f(0,1,0);
    ::glVertex3f(0,  0 ,0); // e0
    ::glVertex3f(0,1.1f,0); // e2

    ::glColor3f(0,1,0);
    ::glVertex3f(1,0,0); // e1
    ::glVertex3f(1,1,0); // e3

    ::glColor3f(0,1,0);
    ::glVertex3f(1,0,1); // e5
    ::glVertex3f(1,1,1); // e7

    ::glColor3f(0,1,0);
    ::glVertex3f(0,0,1); // e4
    ::glVertex3f(0,1,1); // e6

/*---------------------------------------------------------------*/

    ::glColor3f(0,0,1);
    ::glVertex3f(0,0,  0 ); // e0
    ::glVertex3f(0,0,1.1f); // e4

    ::glColor3f(0,0,1);
    ::glVertex3f(1,0,0); // e1
    ::glVertex3f(1,0,1); // e5

    ::glColor3f(0,0,1);
    ::glVertex3f(1,1,0); // e3
    ::glVertex3f(1,1,1); // e7

    ::glColor3f(0,0,1);
    ::glVertex3f(0,1,0); // e2
    ::glVertex3f(0,1,1); // e6
  }
  ::glEnd();
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// mark_vertex
void mark_vertex( char* conf )
//-----------------------------------------------------------------------------
{
  glPointSize( 10.0f) ;
  glColor4f( 0.0f, 1.0f, 0.0f, 1.0f ) ;
  glBegin(GL_POINTS) ;
  {
    if( conf[0] ) glVertex3f(0.0f , 0.0f , 0.0f) ;
    if( conf[1] ) glVertex3f(1.0f , 0.0f , 0.0f) ;
    if( conf[2] ) glVertex3f(1.0f , 1.0f , 0.0f) ;
    if( conf[3] ) glVertex3f(0.0f , 1.0f , 0.0f) ;
    if( conf[4] ) glVertex3f(0.0f , 0.0f , 1.0f) ;
    if( conf[5] ) glVertex3f(1.0f , 0.0f , 1.0f) ;
    if( conf[6] ) glVertex3f(1.0f , 1.0f , 1.0f) ;
    if( conf[7] ) glVertex3f(0.0f , 1.0f , 1.0f) ;
  }
  glEnd() ;
}
//_____________________________________________________________________________




//_____________________________________________________________________________
// mark_face
void mark_face( char f )
//-----------------------------------------------------------------------------
{
  glLightModeli( GL_LIGHT_MODEL_TWO_SIDE,GL_FALSE );
  glColor4f( 1.0f, 0.0f, 0.0f, 0.1f ) ;
  glBegin(GL_QUADS) ;
  {
    switch( f )
    {
    case 1 : glNormal3f(0.0f, -1.0f,  0.0f) ;
      glVertex3f(0.4f , 0.0f , 0.4f) ;
      glVertex3f(0.6f , 0.0f , 0.4f) ;
      glVertex3f(0.6f , 0.0f , 0.6f) ;
      glVertex3f(0.4f , 0.0f , 0.6f) ;
      break ;
    case 2 : glNormal3f(1.0f,  0.0f,  0.0f) ;
      glVertex3f(1.0f , 0.4f , 0.4f) ;
      glVertex3f(1.0f , 0.6f , 0.4f) ;
      glVertex3f(1.0f , 0.6f , 0.6f) ;
      glVertex3f(1.0f , 0.4f , 0.6f) ;
      break ;
    case 3 : glNormal3f(0.0f,  1.0f,  0.0f) ;
      glVertex3f(0.6f , 1.0f , 0.4f) ;
      glVertex3f(0.4f , 1.0f , 0.4f) ;
      glVertex3f(0.4f , 1.0f , 0.6f) ;
      glVertex3f(0.6f , 1.0f , 0.6f) ;
      break ;
    case 4 : glNormal3f(-1.0f, 0.0f,  0.0f) ;
      glVertex3f(0.0f , 0.6f , 0.4f) ;
      glVertex3f(0.0f , 0.6f , 0.6f) ;
      glVertex3f(0.0f , 0.4f , 0.6f) ;
      glVertex3f(0.0f , 0.4f , 0.4f) ;
      break ;
    case 5 : glNormal3f(0.0f,  0.0f, -1.0f) ;
      glVertex3f(0.4f , 0.4f , 0.0f) ;
      glVertex3f(0.6f , 0.4f , 0.0f) ;
      glVertex3f(0.6f , 0.6f , 0.0f) ;
      glVertex3f(0.4f , 0.6f , 0.0f) ;
      break ;
    case 6 : glNormal3f(0.0f,  0.0f,  1.0f ) ;
      glVertex3f(0.4f , 0.4f , 1.0f) ;
      glVertex3f(0.6f , 0.4f , 1.0f) ;
      glVertex3f(0.6f , 0.6f , 1.0f) ;
      glVertex3f(0.4f , 0.6f , 1.0f) ;
      break ;
    }
  }
  glEnd() ;

  // mark interior
  if( f == 7 )
  {
    glBegin(GL_TRIANGLE_STRIP) ;
    {
      glNormal3f( 0.8164970f, 0.4714050f, 0.3333333f) ;
      glVertex3f( 0.4000000f, 0.4000000f, 0.4000000f) ;
      glNormal3f(-0.8164970f, 0.4714050f, 0.3333333f) ;
      glVertex3f( 0.6000000f, 0.4000000f, 0.4000000f) ;
      glNormal3f( 0.0000000f,-0.9428090f, 0.3333333f) ;
      glVertex3f( 0.4100000f, 0.5732051f, 0.4000000f) ;
      glNormal3f( 0.0000000f,-0.0000000f,-1.0000000f) ;
      glVertex3f( 0.4100000f, 0.4577350f, 0.5632993f) ;
      glNormal3f( 0.8164970f, 0.4714050f, 0.3333333f) ;
      glVertex3f( 0.4000000f, 0.4000000f, 0.4000000f) ;
      glNormal3f(-0.8164970f, 0.4714050f, 0.3333333f) ;
      glVertex3f( 0.6000000f, 0.4000000f, 0.4000000f) ;
    }
    glEnd() ;
  }
  glLightModeli( GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE );
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Open GL
void glInit()
//-----------------------------------------------------------------------------
{
  // Default mode
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glShadeModel(GL_FLAT);

  // Lights, material properties
  GLfloat ambient [4] = {0.7f, 0.7f, 0.7f, 1.0f};
  GLfloat diffuse [4] = {0.8f, 0.8f, 0.8f, 1.0f};
  GLfloat specular[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  glClearDepth( 1.0 );
  glClearColor ( 1.0, 1.0, 1.0, 0.0 );

  // Default : lighting
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);

  glLightfv( GL_LIGHT0, GL_AMBIENT , ambient  );
  glLightfv( GL_LIGHT0, GL_DIFFUSE , diffuse  );
  glLightfv( GL_LIGHT0, GL_SPECULAR, specular );
  glLightfv( GL_LIGHT0, GL_POSITION, specular );

  glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER,GL_FALSE );
  glLightModeli( GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE );


  GLfloat shine, emission[4] ;

  ambient [0] = 0.0f  ; ambient [1] = 0.4f ;  ambient [2] = 0.6f ;
  diffuse [0] = 0.0f  ; diffuse [1] = 0.6f ;  diffuse [2] = 0.9f  ;
  specular[0] = 0.0f  ; specular[1] = 0.6f ;  specular[2] = 0.9f  ;
  emission[0] = emission[1] = emission[2] = 0.0f ;
  ambient [3] = diffuse [3] = specular[3] = emission[3] = 0.8f ;
  shine = 64 ;

  glMaterialfv(GL_FRONT,GL_AMBIENT  ,ambient);
  glMaterialfv(GL_FRONT,GL_DIFFUSE  ,diffuse);
  glMaterialfv(GL_FRONT,GL_SPECULAR ,specular);
  glMaterialfv(GL_FRONT,GL_EMISSION ,emission);
  glMaterialf (GL_FRONT,GL_SHININESS,shine);

  ambient [0] = ambient [2] = 0.0f ; ambient [1] = 0.5f ;
  diffuse [0] = diffuse [1] = 0.0f ; diffuse [2] = 0.1f ;
  specular[0] = specular[1] = 0.0f ; specular[2] = 0.8f ;
  emission[0] = emission[1] = emission[2] = 0.0f ;
  ambient [3] = diffuse [3] = specular[3] = emission[3] = 1.0f ;
  shine = 96 ;

  glMaterialfv(GL_BACK,GL_AMBIENT  ,ambient);
  glMaterialfv(GL_BACK,GL_DIFFUSE  ,diffuse);
  glMaterialfv(GL_BACK,GL_SPECULAR ,specular);
  glMaterialfv(GL_BACK,GL_EMISSION ,emission);
  glMaterialf (GL_BACK,GL_SHININESS,shine);

  glDisable( GL_NORMALIZE );
  glDisable( GL_COLOR_MATERIAL );

  glEnable( GL_BLEND );
  glEnable( GL_DEPTH_TEST );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// mouse
int ncalls ;
GLUI_Rotation  mouse_rot, *objects_rot ;
float view_rotate[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
//-----------------------------------------------------------------------------
void mouse(int button, int button_state, int x, int y )
//-----------------------------------------------------------------------------
{
  if ( button == GLUT_LEFT_BUTTON && button_state == GLUT_DOWN )
  {
    mouse_rot.init_live() ;
    mouse_rot.mouse_down_handler(x,y) ;
  }
  if ( button_state != GLUT_DOWN )
    mouse_rot.mouse_up_handler(x,y,1) ;
  ncalls = 0 ;
  objects_rot->sync_live(0,1) ;
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// motion
GLUI *glui ;
//-----------------------------------------------------------------------------
void motion(int x, int y )
//-----------------------------------------------------------------------------
{
  mouse_rot.glui = glui ;
  mouse_rot.iaction_mouse_held_down_handler(x,y,1);
  mouse_rot.glui = NULL ;
  if( ++ncalls > 10 ) { objects_rot->sync_live(0,1) ;  ncalls = 0 ; }
  glutPostRedisplay() ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
//
int   main_window;
//-----------------------------------------------------------------------------
void idle()
//-----------------------------------------------------------------------------
{
  if ( glutGetWindow() != main_window )
    glutSetWindow(main_window);
  glutPostRedisplay();
}
//_____________________________________________________________________________



//_____________________________________________________________________________
//
void reshape( int x, int y )
//-----------------------------------------------------------------------------
{
  int tx, ty, tw, th;
  GLUI_Master.get_viewport_area( &tx, &ty, &tw, &th );
  glViewport( tx, ty, tw, th );
  mouse_rot.set_w( tw ) ;
  mouse_rot.set_h( th ) ;

  float xy_aspect = (float)tw / (float)th;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective( 45.0, xy_aspect, 0.5, 10. );
  gluLookAt( 0.0f,0.0f,3.0f, 0.0f,0.0f,0.0f, 0.0f,1.0f,0.0f ) ;
  glMatrixMode ( GL_MODELVIEW );
  glutPostRedisplay();
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// display
int   wireframe = 0;
int   fill      = 1;
float obj_pos[] = { 0.0, 0.0, 0.0 };
//-----------------------------------------------------------------------------
void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glLoadIdentity();
  glTranslatef( obj_pos[0], obj_pos[1], -obj_pos[2] );
  glMultMatrixf( view_rotate );
  glTranslatef( -0.5f, -0.5f, -0.5f ) ;

  // Axis
/*
  glLineWidth( 1.0f ) ;
  glPushMatrix() ;
  glTranslatef( -0.1f, 0.0f, -0.1f ) ;
  glBegin(GL_LINES) ;
  {
  glColor4f( 0.5f, 0.0f, 0.0f, 1.0f ) ;
  glVertex3f( 0.0f, 0.0f, 0.0f ) ;
  glVertex3f( 0.3f, 0.0f, 0.0f ) ;

  glColor4f( 0.0f, 0.5f, 0.0f, 1.0f ) ;
  glVertex3f( 0.0f, 0.0f, 0.0f ) ;
  glVertex3f( 0.0f, 0.3f, 0.0f ) ;

  glColor4f( 0.0f, 0.0f, 0.5f, 1.0f ) ;
  glVertex3f( 0.0f, 0.0f, 0.0f ) ;
  glVertex3f( 0.0f, 0.0f, 0.3f ) ;
  }
  glEnd() ;
  glPopMatrix() ;
*/

  draw_cube() ;
  glEnable(GL_LIGHTING);
  if( fill )
  {
    // Start rendering...
    if(glIsList(gllist)==GL_TRUE)
      glCallList(gllist);
  }

  glDisable(GL_LIGHTING);
  if( wireframe )
  {
    glPolygonOffset( 0.5, -0.1f );
    glLineWidth(2.0) ;
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE ) ;

    glColor3f(1,1,1) ;
    if(glIsList(gllist)==GL_TRUE)
      glCallList(gllist);

    glLineWidth(1.0) ;
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  }

  // Double buffer
  glutSwapBuffers();
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// Call Back
GLUI_Spinner *case_spin    ;
GLUI_Spinner *subcase_spin ;
GLUI_Spinner *config_spin  ;
//-----------------------------------------------------------------------------
void control_cb( int control )
//-----------------------------------------------------------------------------
{
  switch( control )
  {
  case 0 :
    subcase_spin->set_int_limits( 0, subcasesizes[m_case]-1 ) ;
    config_spin ->set_int_limits( 0, confsizes   [m_case]-1 ) ;
    break ;

  case 1 :
    break ;

  case 2 :
    break ;
  }
  glBuildList() ;
  glutPostRedisplay();
}
//_____________________________________________________________________________


//_____________________________________________________________________________
//
void keybrd(int key, int x, int y)
//-----------------------------------------------------------------------------
{
  switch(key)
  {
  case GLUT_KEY_LEFT      : if( m_config  <= 0 ) m_config  = confsizes   [m_case]-1 ;  else --m_config  ;  break ;
  case GLUT_KEY_RIGHT     : if( m_config  >= confsizes   [m_case]-1 ) m_config  = 0 ;  else ++m_config  ;  break ;
  case GLUT_KEY_DOWN      : if( m_subcase <= 0 ) m_subcase = subcasesizes[m_case]-1 ;  else --m_subcase ;  break ;
  case GLUT_KEY_UP        : if( m_subcase >= subcasesizes[m_case]-1 ) m_subcase = 0 ;  else ++m_subcase ;  break ;
  case GLUT_KEY_PAGE_DOWN : if( m_case    <=  0 ) m_case = 14 ;  else --m_case    ;  break ;
  case GLUT_KEY_PAGE_UP   : if( m_case    >= 14 ) m_case =  0 ;  else ++m_case    ;  break ;
  case GLUT_KEY_HOME      : m_case =  0 ; m_subcase = 0 ; m_config =  0 ;  break ;
  case GLUT_KEY_END       : m_case = 14 ; m_subcase = 0 ; m_config = 11 ;  break ;
  case GLUT_KEY_INSERT    : exit(1) ;  break ;
  }

  printf( "%d, %d, %d\n", m_case, m_subcase, m_config ) ;
  case_spin    ->sync_live(0,1) ;
  subcase_spin ->sync_live(0,1) ;
  config_spin  ->sync_live(0,1) ;

  glBuildList() ;
  glutPostRedisplay() ;
}
//_____________________________________________________________________________




//_____________________________________________________________________________
//
int main (int argc, char **argv)
//-----------------------------------------------------------------------------
{
  m_case    = 0 ;
  m_subcase = 0 ;
  m_config  = 0 ;

  /****************************************/
  /*   Initialize GLUT and create window  */
  /****************************************/

  glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
  glutInitWindowPosition( 50, 50 );
  glutInitWindowSize( 800, 600 );

  /****************************************/
  /*         Here's the GLUI code         */
  /****************************************/

  /*** Create the bottom subwindow ***/
  GLUI_Translation *trans;

  main_window = glutCreateWindow( "Marching Cubes Examples" );
  glui = GLUI_Master.create_glui_subwindow( main_window, GLUI_SUBWINDOW_BOTTOM );
  glui->set_main_gfx_window( main_window );

  objects_rot = glui->add_rotation( "Objects", view_rotate );
  objects_rot->set_spin( 1.0f );
  glui->add_column( false );
  trans = glui->add_translation( "Objects XY", GLUI_TRANSLATION_XY, obj_pos );
  trans->set_speed( .005f );
  glui->add_column( false );
  trans = glui->add_translation( "Objects Z", GLUI_TRANSLATION_Z, &obj_pos[2] );
  trans->set_speed( .005f );
  glui->add_column( true );

  /*** Case selection ***/
  case_spin = glui->add_spinner( "Case", GLUI_SPINNER_INT, &m_case, 0, control_cb ) ;
  case_spin->set_int_limits( 0, 14 ) ;
  subcase_spin = glui->add_spinner( "Subcase", GLUI_SPINNER_INT, &m_subcase, 1, control_cb ) ;
  subcase_spin->set_int_limits( 0, 49 ) ;
  config_spin = glui->add_spinner( "Config", GLUI_SPINNER_INT, &m_config, 2, control_cb ) ;
  config_spin->set_int_limits( 0, 47 ) ;
  glui->add_column( true );

  /*** Wireframe ***/
  glui->add_checkbox( "Fill", &fill, -1, control_cb );
  glui->add_checkbox( "Wireframe", &wireframe, -1, control_cb );
  glui->add_statictext( "" );

  /****** A 'quit' button *****/
  glui->add_button( "Quit", 0,(GLUI_Update_CB)exit );

  /**** Link windows to GLUI, and register idle callback ******/
  glui->set_main_gfx_window( main_window );

  /**** We register the idle callback with GLUI, *not* with GLUT ****/
  GLUI_Master.set_glutIdleFunc( idle );
  GLUI_Master.set_glutReshapeFunc( reshape );
  glutDisplayFunc( display );
  glutMotionFunc ( motion  );
  glutMouseFunc  ( mouse   );
  glutSpecialFunc( keybrd  );

  /**** Init Trackball ****/
  int tx,ty,tw,th ;
  GLUI_Master.get_viewport_area( &tx, &ty, &tw, &th );
  mouse_rot.set_spin(0.05f) ;
  mouse_rot.set_w( tw ) ;
  mouse_rot.set_h( th ) ;
  mouse_rot.set_ptr_val( view_rotate );
  mouse_rot.init_live() ;

  glInit() ;
  glBuildList() ;
  glutMainLoop();

  return 0;
}
//_____________________________________________________________________________

