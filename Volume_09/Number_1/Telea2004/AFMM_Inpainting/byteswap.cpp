#include "byteswap.h"


#ifdef IS_BIG_ENDIAN

void swap2BERange(char*, int) { }
void swap4BERange(char*, int) { }

#else

void swap2BERange(char* mem_ptr1, int num)
{
  char one_byte;
  char *pos;
  int i;
  
  pos = mem_ptr1;
  
  for (i = 0; i < num; i++)
  {
    one_byte = pos[0];
    pos[0] = pos[1];
    pos[1] = one_byte;
    pos = pos + 2;
  }
  
}

void swap4BERange(char* mem_ptr1, int num)
{
  char one_byte;
  char *pos;
  int i;
  
  pos = mem_ptr1;
  
  for (i = 0; i < num; i++)
  {
    one_byte = pos[0];
    pos[0] = pos[3];
    pos[3] = one_byte;
    
    one_byte = pos[1];
    pos[1] = pos[2];
    pos[2] = one_byte;
    pos = pos + 4;
  }
  
}

#endif


