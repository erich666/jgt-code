#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../include/datatype.h"

#include "../include/moller97.c"
//#include "../include/tri_tri.c"
#include "../include/stopwatch.h"

int Linear(float V0[3], float V1[3], float V2[3])
{
  float N[3], E1[3], E2[3];
  SUB(E1,V1,V0);
  SUB(E2,V2,V0);
  CROSS(N,E1,E2);
  if ( (FABS(N[0])>EPSILON) || (FABS(N[1])>EPSILON) || (FABS(N[2])>EPSILON) )
    return 0;
  return 1;
}

void Generate()
{
  int nCountIntersected = 0;
  int nCountSeparated = 0;
  int i;
  int nResult;
  FILE * lf1 = fopen("../data/intersected.txt", "w");
  if (lf1 == NULL)
  {
    printf("Can't generate the data file \"../data/intersected.txt\"!\n");
    return;
  }
  FILE * lf2 = fopen("../data/separated.txt", "w");
  if (lf2 == NULL)
  {
    printf("Can't generate the data file \"../data/separated.txt\"!\n");
    fclose(lf1);
    return;
  }

  float v0[3], v1[3], v2[3];
  float u0[3], u1[3], u2[3];
  srand((unsigned)time(NULL));

  while ( (nCountIntersected<PAIR_NUMBER)
        ||(nCountSeparated  <PAIR_NUMBER) )
  {
    do
    {
      for (i=0; i<3; i++)
      {
        v0[i] = ((float)rand())/RAND_MAX;
        v1[i] = ((float)rand())/RAND_MAX;
        v2[i] = ((float)rand())/RAND_MAX;
      }
    }
    while (Linear(v0, v1, v2)==1);
    do
    {
      for (i=0; i<3; i++)
      {
        u0[i] = ((float)rand())/RAND_MAX;
        u1[i] = ((float)rand())/RAND_MAX;
        u2[i] = ((float)rand())/RAND_MAX;
      }
    }
    while (Linear(u0, u1, u2)==1);

    nResult = tri_tri_intersect(v0, v1, v2, u0, u1, u2);
    if (nResult == 1)
    {
      if (nCountIntersected < PAIR_NUMBER)
      {
        fprintf(lf1, "%.6f %.6f %.6f\n", v0[0], v0[1], v0[2]);
        fprintf(lf1, "%.6f %.6f %.6f\n", v1[0], v1[1], v1[2]);
        fprintf(lf1, "%.6f %.6f %.6f\n", v2[0], v2[1], v2[2]);
        fprintf(lf1, "%.6f %.6f %.6f\n", u0[0], u0[1], u0[2]);
        fprintf(lf1, "%.6f %.6f %.6f\n", u1[0], u1[1], u1[2]);
        fprintf(lf1, "%.6f %.6f %.6f\n", u2[0], u2[1], u2[2]);
        nCountIntersected++;
      }
    }
    else
    {
      if (nCountSeparated < PAIR_NUMBER)
      {
        fprintf(lf2, "%.6f %.6f %.6f\n", v0[0], v0[1], v0[2]);
        fprintf(lf2, "%.6f %.6f %.6f\n", v1[0], v1[1], v1[2]);
        fprintf(lf2, "%.6f %.6f %.6f\n", v2[0], v2[1], v2[2]);
        fprintf(lf2, "%.6f %.6f %.6f\n", u0[0], u0[1], u0[2]);
        fprintf(lf2, "%.6f %.6f %.6f\n", u1[0], u1[1], u1[2]);
        fprintf(lf2, "%.6f %.6f %.6f\n", u2[0], u2[1], u2[2]);
        nCountSeparated++;
      }
    }
  }

  fclose(lf2);
  fclose(lf1);

  printf("Success generating test data sets.\n");
}

void main()
{
  Generate();
}