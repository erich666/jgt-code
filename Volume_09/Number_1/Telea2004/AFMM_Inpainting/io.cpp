#include <math.h>
#include "io.h"
#include "genrl.h"



void float2rgb(float& value,float& R,float& G,float& B)	//simple color-coding routine
{
   const float dx=0.8;

   value = (6-2*dx)*value+dx;
   R = MAX(0,(3-fabs(value-4)-fabs(value-5))/2);
   G = MAX(0,(4-fabs(value-2)-fabs(value-4))/2);
   B = MAX(0,(3-fabs(value-1)-fabs(value-2))/2);
}

		
