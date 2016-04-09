#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../include/datatype.h"

// switch to test moller's method or ours
#if TEST_MOLLER97==1
#include "../include/moller97.c"
#else
#include "../include/tri_tri.c"
#endif

#include "../include/stopwatch.h"

void Test()
{
  float v0[PAIR_NUMBER][3], v1[PAIR_NUMBER][3], v2[PAIR_NUMBER][3];
  float u0[PAIR_NUMBER][3], u1[PAIR_NUMBER][3], u2[PAIR_NUMBER][3];
  int i, j;
  int nResult, nNumber;
  int nCount = 0;
  __int64 time;

  // Load test data
  FILE * lf1 = fopen("../data/intersected.txt", "r");
  if (lf1 == NULL)
  {
    printf("Make sure if the data sets has been generated successfully.");
    return;
  }
  FILE * lf2 = fopen("../data/separated.txt", "r");
  if (lf1 == NULL)
  {
    printf("Make sure if the data sets has been generated successfully.");
    fclose(lf1);
    return;
  }

  nNumber = PAIR_NUMBER * INTERSECTION_RATIO;
  for (i=0; i<nNumber; i++)
  {
    fscanf(lf1, "%f %f %f", &v0[i][0], &v0[i][1], &v0[i][2]);
    fscanf(lf1, "%f %f %f", &v1[i][0], &v1[i][1], &v1[i][2]);
    fscanf(lf1, "%f %f %f", &v2[i][0], &v2[i][1], &v2[i][2]);
    fscanf(lf1, "%f %f %f", &u0[i][0], &u0[i][1], &u0[i][2]);
    fscanf(lf1, "%f %f %f", &u1[i][0], &u1[i][1], &u1[i][2]);
    fscanf(lf1, "%f %f %f", &u2[i][0], &u2[i][1], &u2[i][2]);
  }
  for (i=nNumber; i<PAIR_NUMBER; i++)
  {
    fscanf(lf2, "%f %f %f", &v0[i][0], &v0[i][1], &v0[i][2]);
    fscanf(lf2, "%f %f %f", &v1[i][0], &v1[i][1], &v1[i][2]);
    fscanf(lf2, "%f %f %f", &v2[i][0], &v2[i][1], &v2[i][2]);
    fscanf(lf2, "%f %f %f", &u0[i][0], &u0[i][1], &u0[i][2]);
    fscanf(lf2, "%f %f %f", &u1[i][0], &u1[i][1], &u1[i][2]);
    fscanf(lf2, "%f %f %f", &u2[i][0], &u2[i][1], &u2[i][2]);
  }

  fclose(lf2);
  fclose(lf1);


 
  
  // Test performance
  CStopwatch watch;
  for (j=0; j<LOOP_NUMBER; j++)
  {
    for (i=0; i<PAIR_NUMBER; i++)
    {
      nResult = tri_tri_intersect(u0[i], u1[i], u2[i], v0[i], v1[i], v2[i]);
      nCount+=nResult;
    }
  }
  time = watch.Now(MICROSECOND);
  printf("Number of loops : %d\nNumber of pairs : %d\nRatio of intersection : %.6f\nIntersected pair number : %d\nCost time : %d microseconds\n",
    LOOP_NUMBER, PAIR_NUMBER, INTERSECTION_RATIO, nCount/LOOP_NUMBER, time);
}

void main()
{
  Test();
}