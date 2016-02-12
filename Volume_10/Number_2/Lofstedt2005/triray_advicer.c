/* 

triray_advicer.c

TODO

FIXA KOMMENTARER TILL dvicer_algorithms. Måste ha med litteratur referennser och vad jag hittat på själv ifråga om optimeringar


*/
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "triangle.h"
#include "advicer_constants.h"



void num_sumresults(float *r, float start, float stop)
{
  int i,j;
  for(i=(int) 10*start;i <= (int)10*stop ;i++){
  
    for(j=0;j<(NR_ALGOS);j++)
      r[NR_ALGOS*(NR_SCALES+1)+j]+=r[i*NR_ALGOS+j];
  }

}


void printval(algo_val * r)
{
  
    switch (r-> algo) {
    case  0:
      printf("MT0");
      break;
    case  1:
      printf("MT1");
      break;
    case  2:
      printf("MT2");
      break;
    case  3:
      printf("MT3");
      break;
    case  4:
      printf("MA");
      break;
    case  5:
      printf("PU");
      break;
    case  6:
      printf("OR");
      break;
    case  7:
      printf("ORC");
      break;
    case  8:
      printf("MApl");
      break;
    case  9:
      printf("PUpl");
      break;
    case  10:
      printf("CH1p");
      break;
    case  11:
      printf("CH2p");
      break;
    case  12:
      printf("CH3p");
      break;
    case  13:
      printf("ORp");
      break;
    case  14:
      printf("ORCp");
      break;
    case  15:
      printf("HFp");
      break;
    case  16:
      printf("HF2p");
      break;
    case  17:
      printf("A2Dp");
      break;
    case  18:
      printf("HFh");
      break;
    case  19:
      printf("HF2h");
      break;
    case  20:
      printf("ARi");
      break;
    case  21:
      printf("AR2i");
      break;
    }
}


void mkAlgoVal(float v,int i,algo_val *r){
  r->value=v;
  r->algo=i;

}

int comp(const void *x,const void *y)
{
  const float *a = x;
  const float *b = y;  
  return ((*a >= *b) ? ((*a > *b) ? -1 : 0):1);
}

int comp_val(const void *x,const void *y)
{
  const algo_val *a = x;
  const algo_val *b = y;  
  return ((a->value <= b->value) ? ((a->value < b->value) ? -1 : 0):1);
}



int main(int argc, char *argv[])
{
  
  int a;
  int m=0;
  float start_h = 0, stop_h = 0.1*(NR_SCALES);
  
  float cps = (float) CLOCKS_PER_SEC * NR_SETS;
  
  float orig[3] = {0.,0.,0.};
  float dir[3] = {0.,0.,0.};
  float end[3] = {0.,0.,0.};
  
  float vert0[3] = {0,0,0};
  float vert1[3] = {0,0,0};
  float vert2[3] = {0,0,0};
  float tmp[3] = {0,0,0};
  float tmp2[3] = {0,0,0};
  float tmpn[3] = {0,0,0};
  
  float *t =  (float *)malloc(sizeof(float)); 
  float *u =  (float *)malloc(sizeof(float));  
  float *v =  (float *)malloc(sizeof(float)); 
  float point[3] = {0,0,0};

  float *results_bary =  (float*)malloc(sizeof(float)*NR_ALGOS*(NR_SCALES+2));
  float *results_t =  (float*)malloc(sizeof(float)*NR_ALGOS*(NR_SCALES+2));
  float *results =  (float*)malloc(sizeof(float)*NR_ALGOS*(NR_SCALES+2));
  
  
  Ray_big *rays = (Ray_big *) malloc(sizeof(Ray_big)*N);
  Triangle_small *tris2;
  Plucker_coords *tris3;
  Triangle_plane *tris4;
  
  Triangle_Halfplane *tris5;
  Triangle_inv *tris6;
  
  FILE * outfile, * texfile;
  
  Intersection_big *inters;
  
  int i,j,k,l,hit,hits_so_far=0, miss_so_far=0,nr_hits;
  float v_temp=0.;
  int t0_hits=0,t1_hits=0,t2_hits=0,t3_hits=0,b0_hits=0,s0_hits=0,c0_hits=0,p0_hits=0,cr0_hits=0,pl0_hits=0,su0_hits=0,c1_hits=0,e0_hits=0,m0_hits=0, b1_hits=0, t01_hits=0, p1_hits=0, t02_hits=0, o0_hits=0, h0_hits=0, h1_hits=0, a122_hits=0, a2D_hits=0, sn_hits=0, occw_hits=0, ar_hits=0, cr3_hits=0, ar2_hits=0;
  unsigned long t0_time=0,t1_time=0,t2_time=0,t3_time=0,b0_time=0,s0_time=0,c0_time=0,p0_time=0,cr0_time=0,pl0_time=0,su0_time=0, c1_time=0, e0_time=0,m0_time=0, b1_time=0, t01_time=0, p1_time=0, t02_time=0, o0_time=0, h0_time=0, h1_time=0, a122_time=0, a2D_time=0, sn_time=0, occw_time=0, ar_time=0, cr3_time=0, ar2_time=0;
  
  unsigned long start_time = 0;
  
  int reps =  0;
  algo_val  *test_res=(algo_val*)malloc(sizeof(algo_val)*NR_ALGOS);
  
  srand(time(NULL));
  
  nr_hits=0;

 
   if(argc == 3){
   
    sscanf(argv[1], "%f", &start_h);
    sscanf(argv[2], "%f", &stop_h); 
   
   }
   
   /* If no arguments are entered or arguments in wrong range we run the hole range i.e we start at hitrate 0.0 and end at hitrate 1.0*/
   if(stop_h < start_h || stop_h > NR_SCALES || stop_h < 0.0 || stop_h > 1.0 || start_h < 0.0 || start_h > 1.0){
     start_h=0.;
     stop_h=0.1*(NR_SCALES);
   }
   
   
   /********************************************************************
Pass 1
 Test on the small triangle struct, everything is calculated on the fly

   *********************************************************************/
  tris2 = (Triangle_small *) malloc(sizeof(Triangle_small )*N);
  
 

  hits_so_far = 0;
  miss_so_far =0;
  
  nr_hits=start_h*N;
  
  while(nr_hits <= stop_h*N){
    /*    printf("nr_hits: %d, \n", nr_hits ); */
    for(j =0;j<NR_SETS;j++){
      start_time = clock(); 
      for(i = 0; i < N; i++)
	{
	  
	  /* set origin to a point at the unit sphere*/
	  orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    orig[2] = (float)(sqrt(1. - v_temp));
	  else
	    orig[2] = - (float)(sqrt(1. - v_temp));
	  
	  /*Set end to a point at the unit sphere*/
	  end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  end[0]*end[0] + end[1]*end[1];
	  /* make end ok fore square root and different fron orig*/
	  while(v_temp > 1. || (end[0] == orig[0] && end[1] == orig[1] )){
	    end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  end[0]*end[0] + end[1]*end[1];
	  }
	  
	  
	  if(rand() > RAND_MAX/2)
	    end[2] = (float)(sqrt(1. - v_temp));
	  else
	    end[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  dir[0] = end[0] - orig[0];
	  dir[1] = end[1] - orig[1];
	  dir[2] = end[2] - orig[2];
	  
	  NORMALIZE(dir);
	  
	  
	  /* set origin to a point at the unit sphere*/
	  vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert0[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert0[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  /* set origin to a point at the unit sphere*/
	  vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert1[0] == vert0[0])
	    {
	    vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	    }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert1[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert1[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  /* set origin to a point at the unit sphere*/
	  vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert2[0] == vert1[0] || vert2[0] == vert0[0]){
	    vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert2[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert2[2] = - (float)(sqrt(1. - v_temp));
	  
	  /*
	    Test if we have a hit and we need more hits or 
	    if we dont have a hit and dont need anymore
	    add ray and triangle to lists
	  */
	  hit = test_hit(orig, dir,vert0, vert1, vert2); 
	  
	  if( (hit && hits_so_far < nr_hits) || (!hit && (miss_so_far <= (N - nr_hits -1)))){
	    mkRay_big(orig, dir, end, &rays[i]);
	    if(hit)
	      hits_so_far++;
	    else
	      miss_so_far++;
	    SUB(tmp, vert1, vert0);
	    SUB(tmp2, vert2, vert0);
	    CROSS(tmpn, tmp,tmp2);
	    NORMALIZE(tmpn);
	    
	    if(DOT(dir,tmpn) > 0.){
	      tmp[0] = vert0[0];
	      tmp[1] = vert0[1];
	      tmp[2] = vert0[2];
	      vert0[0] = vert2[0];
	      vert0[1] = vert2[1];
	      vert0[2] = vert2[2];
	      vert2[0] = tmp[0];
	      vert2[1] = tmp[1];
	      vert2[2] = tmp[2];
	    }
	    mkTriangle_small(vert0, vert1, vert2, &tris2[i]);
	  }
	  else
	    i--;    
	}
      inters = mkIntersection_big(t,u,v,point);
    
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t0_hits+=intersect_triangle_small_bary(&rays[i], &tris2[i], inters);
      t0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + mt0]=((float)t0_time)/cps;
   
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t1_hits+=intersect_triangle1_small_bary(&rays[i], &tris2[i], inters);
      t1_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + mt1]=((float)t1_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t2_hits+=intersect_triangle2_small_bary(&rays[i], &tris2[i], inters);
      t2_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + mt2]=((float)t2_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t3_hits+=intersect_triangle3_small_bary(&rays[i], &tris2[i], inters);
      t3_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + mt3]=((float)t3_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  m0_hits+=plucker_mahovsky_small_bary(&rays[i], &tris2[i], inters);
      m0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + ma]=((float)m0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  pl0_hits+=plucker_small_bary(&rays[i], &tris2[i], inters);
      pl0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + pu]=((float)pl0_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS;k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  o0_hits+=orourke_small_bary(&rays[i], &tris2[i], inters);
      o0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + or]=((float)o0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS;k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  occw_hits+=orourke_small_baryCCW(&rays[i], &tris2[i], inters);
      occw_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + orc]=((float)occw_time)/cps;
  
    t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
    t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
      /*T-value*/
         start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t0_hits+=intersect_triangle_small_t(&rays[i], &tris2[i], inters);
      t0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + mt0]=((float)t0_time)/cps;


       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t1_hits+=intersect_triangle1_small_t(&rays[i], &tris2[i], inters);
      t1_time += clock() - start_time;
      results_t[reps*NR_ALGOS + mt1]=((float)t1_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t2_hits+=intersect_triangle2_small_t(&rays[i], &tris2[i], inters);
      t2_time += clock() - start_time;
      results_t[reps*NR_ALGOS + mt2]=((float)t2_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t3_hits+=intersect_triangle3_small_t(&rays[i], &tris2[i], inters);
      t3_time += clock() - start_time;
      results_t[reps*NR_ALGOS + mt3]=((float)t3_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  m0_hits+=plucker_mahovsky_small_t(&rays[i], &tris2[i], inters);
      m0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + ma]=((float)m0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  pl0_hits+=plucker_small_t(&rays[i], &tris2[i], inters);
      pl0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + pu]=((float)pl0_time)/cps;
  
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS;k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  o0_hits+=orourke_small_t(&rays[i], &tris2[i], inters);
      o0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + or]=((float)o0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS;k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  occw_hits+=orourke_small_tCCW(&rays[i], &tris2[i], inters);
      occw_time += clock() - start_time;
      results_t[reps*NR_ALGOS + orc]=((float)occw_time)/cps;
      
      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
        /*BOOL*/
        start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t0_hits+=intersect_triangle_small(&rays[i], &tris2[i], inters);
      t0_time += clock() - start_time;
      results[reps*NR_ALGOS + mt0]=((float)t0_time)/cps;
   
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t1_hits+=intersect_triangle1_small(&rays[i], &tris2[i], inters);
      t1_time += clock() - start_time;
      results[reps*NR_ALGOS + mt1]=((float)t1_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t2_hits+=intersect_triangle2_small(&rays[i], &tris2[i], inters);
      t2_time += clock() - start_time;
      results[reps*NR_ALGOS + mt2]=((float)t2_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  t3_hits+=intersect_triangle3_small(&rays[i], &tris2[i], inters);
      t3_time += clock() - start_time;
      results[reps*NR_ALGOS + mt3]=((float)t3_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  m0_hits+=plucker_mahovsky_small(&rays[i], &tris2[i], inters);
      m0_time += clock() - start_time;
      results[reps*NR_ALGOS + ma]=((float)m0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  pl0_hits+=plucker_small(&rays[i], &tris2[i], inters);
      pl0_time += clock() - start_time;
      results[reps*NR_ALGOS + pu]=((float)pl0_time)/cps;
  
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS;k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  o0_hits+=orourke_small(&rays[i], &tris2[i], inters);
      o0_time += clock() - start_time;
      results[reps*NR_ALGOS + or]=((float)o0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS;k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  occw_hits+=orourke_smallCCW(&rays[i], &tris2[i], inters);
      occw_time += clock() - start_time;
      results[reps*NR_ALGOS + orc]=((float)occw_time)/cps;

      
      hits_so_far = 0;
      miss_so_far = 0;
      
      reps++;
      free(inters);
    }
 
    
    nr_hits += N/NR_SCALES;
    
    t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
    t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
  }
  reps=0;


  /*****************************************************
Pass 2
 Test on Plucker coordinates triangle struct 

  ******************************************************/

  free(tris2);
  tris3 = (Plucker_coords *) malloc(sizeof(Plucker_coords)*N);
 
  hits_so_far = 0;
  miss_so_far = 0;
 
  
  nr_hits=start_h*N;
  
  while(nr_hits <= stop_h*N){
    /*printf("nr_hits: %d, \n", nr_hits );*/
    for(j =0;j<NR_SETS;j++){
      for(i = 0; i < N; i++)
	{
	  orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    orig[2] = (float)(sqrt(1. - v_temp));
	  else
	    orig[2] = - (float)(sqrt(1. - v_temp));
	  
	  /*Set end to a point at the unit sphere*/
	  end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  end[0]*end[0] + end[1]*end[1];
	  /* make end ok fore square root and different fron orig*/
	  while(v_temp > 1. || (end[0] == orig[0] && end[1] == orig[1] )){
	    end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  end[0]*end[0] + end[1]*end[1];
	  }

	  	  
	  if(rand() > RAND_MAX/2)
	    end[2] = (float)(sqrt(1. - v_temp));
	  else
	    end[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  dir[0] = end[0] - orig[0];
	  dir[1] = end[1] - orig[1];
	  dir[2] = end[2] - orig[2];
	  
	  NORMALIZE(dir);

	   /* set origin to a point at the unit sphere*/
	  vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert0[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert0[2] = - (float)(sqrt(1. - v_temp));


	  	   /* set origin to a point at the unit sphere*/
	  vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert1[0] == vert0[0]){
	    vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert1[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert1[2] = - (float)(sqrt(1. - v_temp));

	  
	  	   /* set origin to a point at the unit sphere*/
	  vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert2[0] == vert1[0] || vert2[0] == vert0[0]){
	    vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert2[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert2[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  
	  /*
	    Test if we have a hit and we need more hits or 
	    if we dont have a hit and dont need anymore
	    add ray and triangle to lists
	  */
	  hit = test_hit(orig, dir,vert0, vert1, vert2); 
	  
	  if( (hit && hits_so_far < nr_hits) || (!hit && (miss_so_far <= (N - nr_hits -1)))){
	    mkRay_big(orig, dir, end, &rays[i]);
	    if(hit)
	      hits_so_far++;
	    else
	      miss_so_far++;
	    SUB(tmp, vert1, vert0);
	    SUB(tmp2, vert2, vert0);
	    CROSS(tmpn, tmp,tmp2);
	    NORMALIZE(tmpn);
	    
	    if(DOT(dir,tmpn) > 0.){
	      tmp[0] = vert0[0];
	      tmp[1] = vert0[1];
	      tmp[2] = vert0[2];
	      vert0[0] = vert2[0];
	      vert0[1] = vert2[1];
	      vert0[2] = vert2[2];
	      vert2[0] = tmp[0];
	      vert2[1] = tmp[1];
	      vert2[2] = tmp[2];
	    }
	    mkPlucker(vert0, vert1, vert2, &tris3[i]);
	  }
	  else
	    i--;    
	}
      inters = mkIntersection_big(t,u,v,point);
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  m0_hits+=plucker_mahovsky_other_bary(&rays[i], &tris3[i], inters);
      m0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + mapl]=((float)m0_time)/cps;
  
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  pl0_hits+=plucker_other_bary(&rays[i], &tris3[i], inters);
      pl0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + pupl]=((float)pl0_time)/cps;

      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
         start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  m0_hits+=plucker_mahovsky_other_t(&rays[i], &tris3[i], inters);
      m0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + mapl]=((float)m0_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  pl0_hits+=plucker_other_t(&rays[i], &tris3[i], inters);
      pl0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + pupl]=((float)pl0_time)/cps;
      
      
      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  m0_hits+=plucker_mahovsky_other(&rays[i], &tris3[i], inters);
      m0_time += clock() - start_time;
      results[reps*NR_ALGOS + mapl]=((float)m0_time)/cps;
  
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  pl0_hits+=plucker_other(&rays[i], &tris3[i], inters);
      pl0_time += clock() - start_time;
      results[reps*NR_ALGOS + pupl]=((float)pl0_time)/cps;

      
      hits_so_far = 0;
      miss_so_far = 0;
      
      reps++;
      
      free(inters);
    }
        
    nr_hits += N/NR_SCALES;
    t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
    t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
  }
  
  reps=0;
  

  /************************************************************
Pass 3
 Test on triangle struct with precalculated plane equation 

  *************************************************************/

  free(tris3);
  tris4 = (Triangle_plane *) malloc(sizeof(Triangle_plane) *N);
 
  hits_so_far = 0;
  miss_so_far = 0;

  nr_hits=start_h*N;
  
  while(nr_hits <= stop_h*N){
    /*printf("nr_hits: %d, \n", nr_hits );*/
    for(j =0;j<NR_SETS;j++){
      for(i = 0; i < N; i++)
	{
	  /* set origin to a point at the unit sphere*/
	  orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    orig[2] = (float)(sqrt(1. - v_temp));
	  else
	    orig[2] = - (float)(sqrt(1. - v_temp));
	  
	  /*Set end to a point at the unit sphere*/
	  end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  end[0]*end[0] + end[1]*end[1];
	  /* make end ok fore square root and different fron orig*/
	  while(v_temp > 1. || (end[0] == orig[0] && end[1] == orig[1] )){
	    end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  end[0]*end[0] + end[1]*end[1];
	  }

	  	  
	  if(rand() > RAND_MAX/2)
	    end[2] = (float)(sqrt(1. - v_temp));
	  else
	    end[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  dir[0] = end[0] - orig[0];
	  dir[1] = end[1] - orig[1];
	  dir[2] = end[2] - orig[2];
	  
	  NORMALIZE(dir);
	 

	   /* set origin to a point at the unit sphere*/
	  vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert0[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert0[2] = - (float)(sqrt(1. - v_temp));


	  	   /* set origin to a point at the unit sphere*/
	  vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert1[0] == vert0[0]){
	    vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert1[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert1[2] = - (float)(sqrt(1. - v_temp));

	  
	  	   /* set origin to a point at the unit sphere*/
	  vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert2[0] == vert1[0] || vert2[0] == vert0[0]){
	    vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert2[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert2[2] = - (float)(sqrt(1. - v_temp));



	  /*
	    Test if we have a hit and we need more hits or 
	    if we dont have a hit and dont need anymore
	    add ray and triangle to lists
	  */
	  hit = test_hit(orig, dir,vert0, vert1, vert2); 
	  
	  if( (hit && hits_so_far < nr_hits) || (!hit && (miss_so_far <= (N - nr_hits -1)))){
	    mkRay_big(orig, dir, end, &rays[i]);
	    if(hit)
	      hits_so_far++;
	    else
	      miss_so_far++;
	    SUB(tmp, vert1, vert0);
	    SUB(tmp2, vert2, vert0);
	    CROSS(tmpn, tmp,tmp2);
	    NORMALIZE(tmpn);
	    
	    if(DOT(dir,tmpn) > 0.){
	      tmp[0] = vert0[0];
	      tmp[1] = vert0[1];
	      tmp[2] = vert0[2];
	      vert0[0] = vert2[0];
	      vert0[1] = vert2[1];
	      vert0[2] = vert2[2];
	      vert2[0] = tmp[0];
	      vert2[1] = tmp[1];
	      vert2[2] = tmp[2];
	      }
	    mkTriangle_plane(vert0, vert1, vert2, &tris4[i]);
	  }
	  else
	    i--;    
	}
      inters = mkIntersection_big(t,u,v,point);
      
      /* no branch */
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  c0_hits+=chirkov_other_bary(&rays[i], &tris4[i], inters);
      c0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + ch1p]=((float)c0_time)/cps;
      
      /*chirkov orig*/
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  c1_hits+=chirkov2_other_bary(&rays[i], &tris4[i], inters);
      c1_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + ch2p]=((float)c1_time)/cps;

      /*orig opt*/
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  cr3_hits+=chirkov3_other_bary(&rays[i], &tris4[i], inters);
      cr3_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + ch3p]=((float)cr3_time)/cps;
   
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  o0_hits+=orourke_other_bary(&rays[i], &tris4[i], inters);
      o0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + orp]=((float)o0_time)/cps;

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  occw_hits+=orourke_other_baryCCW(&rays[i], &tris4[i], inters);
      occw_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + orcp]=((float)occw_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h0_hits+= halfplane_other_bary(&rays[i], &tris4[i], inters);
      h0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + hfp]=((float)h0_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h1_hits+= halfplane2_other_bary(&rays[i], &tris4[i], inters);
      h1_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + hf2p]=((float)h1_time)/cps;

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  a2D_hits+= area2D_other_bary(&rays[i], &tris4[i], inters);
      a2D_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + a2dp]=((float)a2D_time)/cps;


      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  c0_hits+=chirkov_other_t(&rays[i], &tris4[i], inters);
      c0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + ch1p]=((float)c0_time)/cps;
      
      /*chirkov orig*/
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  c1_hits+=chirkov2_other_t(&rays[i], &tris4[i], inters);
      c1_time += clock() - start_time;
      results_t[reps*NR_ALGOS + ch2p]=((float)c1_time)/cps;

      /*orig opt*/
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  cr3_hits+=chirkov3_other_t(&rays[i], &tris4[i], inters);
      cr3_time += clock() - start_time;
      results_t[reps*NR_ALGOS + ch3p]=((float)cr3_time)/cps;
   
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  o0_hits+=orourke_other_t(&rays[i], &tris4[i], inters);
      o0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + orp]=((float)o0_time)/cps;

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  occw_hits+=orourke_other_tCCW(&rays[i], &tris4[i], inters);
      occw_time += clock() - start_time;
      results_t[reps*NR_ALGOS + orcp]=((float)occw_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h0_hits+= halfplane_other_t(&rays[i], &tris4[i], inters);
      h0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + hfp]=((float)h0_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h1_hits+= halfplane2_other_t(&rays[i], &tris4[i], inters);
      h1_time += clock() - start_time;
      results_t[reps*NR_ALGOS + hf2p]=((float)h1_time)/cps;

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  a2D_hits+= area2D_other_t(&rays[i], &tris4[i], inters);
      a2D_time += clock() - start_time;
      results_t[reps*NR_ALGOS + a2dp]=((float)a2D_time)/cps;
      
      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
      /* no branch */
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  c0_hits+=chirkov_other(&rays[i], &tris4[i], inters);
      c0_time += clock() - start_time;
      results[reps*NR_ALGOS + ch1p]=((float)c0_time)/cps;
      
      /*chirkov orig*/
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  c1_hits+=chirkov2_other(&rays[i], &tris4[i], inters);
      c1_time += clock() - start_time;
      results[reps*NR_ALGOS + ch2p]=((float)c1_time)/cps;

      /*orig opt*/
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  cr3_hits+=chirkov3_other(&rays[i], &tris4[i], inters);
      cr3_time += clock() - start_time;
      results[reps*NR_ALGOS + ch3p]=((float)cr3_time)/cps;
   
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  o0_hits+=orourke_other(&rays[i], &tris4[i], inters);
      o0_time += clock() - start_time;
      results[reps*NR_ALGOS + orp]=((float)o0_time)/cps;

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  occw_hits+=orourke_otherCCW(&rays[i], &tris4[i], inters);
      occw_time += clock() - start_time;
      results[reps*NR_ALGOS + orcp]=((float)occw_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h0_hits+= halfplane_other(&rays[i], &tris4[i], inters);
      h0_time += clock() - start_time;
      results[reps*NR_ALGOS + hfp]=((float)h0_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h1_hits+= halfplane2_other(&rays[i], &tris4[i], inters);
      h1_time += clock() - start_time;
      results[reps*NR_ALGOS + hf2p]=((float)h1_time)/cps;

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  a2D_hits+= area2D_other(&rays[i], &tris4[i], inters);
      a2D_time += clock() - start_time;
      results[reps*NR_ALGOS + a2dp]=((float)a2D_time)/cps;

      hits_so_far = 0;
      miss_so_far = 0;
      
      reps++;
      free(inters);
    }
 

    

    nr_hits += N/NR_SCALES;
    t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
    t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
  
  
  }
    reps=0;
  
  

    /************************************************************
 Pass 4
Test on triangle struct with precalculated half-plane equations

    *************************************************************/


  free(tris4);
  tris5 = (Triangle_Halfplane *) malloc(sizeof(Triangle_Halfplane)*N);

  hits_so_far = 0;
  miss_so_far = 0;
  
  nr_hits= start_h*N;
  
  while(nr_hits <= stop_h*N){
    /*printf("nr_hits: %d, \n", nr_hits );*/
    for(j =0;j<NR_SETS;j++){
      for(i = 0; i < N; i++)
	{
	  orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    orig[2] = (float)(sqrt(1. - v_temp));
	  else
	    orig[2] = - (float)(sqrt(1. - v_temp));
	  
	  /*Set end to a point at the unit sphere*/
	  end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  end[0]*end[0] + end[1]*end[1];
	  /* make end ok fore square root and different fron orig*/
	  while(v_temp > 1. || (end[0] == orig[0] && end[1] == orig[1] )){
	    end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  end[0]*end[0] + end[1]*end[1];
	  }

       	  if(rand() > RAND_MAX/2)
	    end[2] = (float)(sqrt(1. - v_temp));
	  else
	    end[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  dir[0] = end[0] - orig[0];
	  dir[1] = end[1] - orig[1];
	  dir[2] = end[2] - orig[2];
	  
	  NORMALIZE(dir);

	   /* set origin to a point at the unit sphere*/
	  vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert0[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert0[2] = - (float)(sqrt(1. - v_temp));

	  /* set origin to a point at the unit sphere*/
	  vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert1[0] == vert0[0]){
	    vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert1[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert1[2] = - (float)(sqrt(1. - v_temp));

	  /* set origin to a point at the unit sphere*/
	  vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert2[0] == vert1[0] || vert2[0] == vert0[0]){
	    vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert2[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert2[2] = - (float)(sqrt(1. - v_temp));

	  /*
	    Test if we have a hit and we need more hits or 
	    if we dont have a hit and dont need anymore
	    add ray and triangle to lists
	  */
	  hit = test_hit(orig, dir,vert0, vert1, vert2); 
	  
	  if( (hit && hits_so_far < nr_hits) || (!hit && (miss_so_far <= (N - nr_hits -1)))){
	    mkRay_big(orig, dir, end, &rays[i]);
	    if(hit)
	      hits_so_far++;
	    else
	      miss_so_far++;
	    SUB(tmp, vert1, vert0);
	    SUB(tmp2, vert2, vert0);
	    CROSS(tmpn, tmp,tmp2);
	    NORMALIZE(tmpn);
	    
	    if(DOT(dir,tmpn) > 0.){
	      tmp[0] = vert0[0];
	      tmp[1] = vert0[1];
	      tmp[2] = vert0[2];
	      vert0[0] = vert2[0];
	      vert0[1] = vert2[1];
	      vert0[2] = vert2[2];
	      vert2[0] = tmp[0];
	      vert2[1] = tmp[1];
	      vert2[2] = tmp[2];
	    }
	    mkTriangle_half(vert0, vert1, vert2, &tris5[i]);
	  }
	  else
	    i--;    
	}
      inters = mkIntersection_big(t,u,v,point);

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h0_hits+=halfplane_other_bary_pre(&rays[i], &tris5[i], inters);
      h0_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + hfh]=((float)h0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h1_hits+=halfplane_other_bary_pre2(&rays[i], &tris5[i], inters);
      h1_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + hf2h]=((float)h1_time)/cps;
      
      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h0_hits+=halfplane_other_t_pre(&rays[i], &tris5[i], inters);
      h0_time += clock() - start_time;
      results_t[reps*NR_ALGOS + hfh]=((float)h0_time)/cps;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h1_hits+=halfplane_other_t_pre2(&rays[i], &tris5[i], inters);
      h1_time += clock() - start_time;
      results_t[reps*NR_ALGOS + hf2h]=((float)h1_time)/cps;
      
      
      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h0_hits+=halfplane_other_pre(&rays[i], &tris5[i], inters);
      h0_time += clock() - start_time;
      results[reps*NR_ALGOS + hfh]=((float)h0_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  h1_hits+=halfplane_other_pre2(&rays[i], &tris5[i], inters);
      h1_time += clock() - start_time;
      results[reps*NR_ALGOS + hf2h]=((float)h1_time)/cps;

    
      hits_so_far = 0;
      miss_so_far = 0;
      reps++;
      
      free(inters);
    }


    
    nr_hits += N/NR_SCALES;
    t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
    t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;


  }
  reps=0;




  /******************************************************
Pass 5
 test on triangles struct with precalculated invers for the Arenberg algorithm

  ******************************************************/


  free(tris5);
  tris6 = (Triangle_inv *) malloc(sizeof(Triangle_inv)*N);
 
  hits_so_far = 0;
  miss_so_far = 0;
  
  nr_hits=start_h*N;
  
  while(nr_hits <= stop_h*N){
    /*printf("nr_hits: %d, \n", nr_hits );*/
    for(j =0;j<NR_SETS;j++){
      for(i = 0; i < N; i++)
	{
	  /* set origin to a point at the unit sphere*/
	  orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    orig[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    orig[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  orig[0]*orig[0] + orig[1]*orig[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    orig[2] = (float)(sqrt(1. - v_temp));
	  else
	    orig[2] = - (float)(sqrt(1. - v_temp));
	  
	  /*Set end to a point at the unit sphere*/
	  end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  end[0]*end[0] + end[1]*end[1];
	  /* make end ok fore square root and different fron orig*/
	  while(v_temp > 1. || (end[0] == orig[0] && end[1] == orig[1] )){
	    end[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    end[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  end[0]*end[0] + end[1]*end[1];
	  }

	  	  
	  if(rand() > RAND_MAX/2)
	    end[2] = (float)(sqrt(1. - v_temp));
	  else
	    end[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  dir[0] = end[0] - orig[0];
	  dir[1] = end[1] - orig[1];
	  dir[2] = end[2] - orig[2];
	  
	  NORMALIZE(dir);

	   /* set origin to a point at the unit sphere*/
	  vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1.){
	    vert0[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert0[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert0[0]*vert0[0] + vert0[1]*vert0[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert0[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert0[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  /* set origin to a point at the unit sphere*/
	  vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert1[0] == vert0[0]){
	    vert1[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert1[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert1[0]*vert1[0] + vert1[1]*vert1[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert1[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert1[2] = - (float)(sqrt(1. - v_temp));
	  
	  
	  /* set origin to a point at the unit sphere*/
	  vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  /*Make shure we have appropriate values for square root*/
	  while(v_temp > 1. || vert2[0] == vert1[0] || vert2[0] == vert0[0]){
	    vert2[0] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    vert2[1] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	    v_temp =  vert2[0]*vert2[0] + vert2[1]*vert2[1];
	  }
	  /* randomize sign of z coord, and pick a point on the unit sphere*/
	  if(rand() > RAND_MAX/2)
	    vert2[2] = (float)(sqrt(1. - v_temp));
	  else
	    vert2[2] = - (float)(sqrt(1. - v_temp));



	  /*
	    Test if we have a hit and we need more hits or 
	    if we dont have a hit and dont need anymore
	    add ray and triangle to lists
	  */
	  hit = test_hit(orig, dir,vert0, vert1, vert2); 
	  
	  if( (hit && hits_so_far < nr_hits) || (!hit && (miss_so_far <= (N - nr_hits -1)))){
	    mkRay_big(orig, dir, end, &rays[i]);
	    if(hit)
	      hits_so_far++;
	    else
	      miss_so_far++;
	    SUB(tmp, vert1, vert0);
	    SUB(tmp2, vert2, vert0);
	    CROSS(tmpn, tmp,tmp2);
	    NORMALIZE(tmpn);
	    
	    if(DOT(dir,tmpn) > 0.){
	      tmp[0] = vert0[0];
	      tmp[1] = vert0[1];
	      tmp[2] = vert0[2];
	      vert0[0] = vert2[0];
	      vert0[1] = vert2[1];
	      vert0[2] = vert2[2];
	      vert2[0] = tmp[0];
	      vert2[1] = tmp[1];
	      vert2[2] = tmp[2];
	    }
	    mkTriangle_inv(vert0, vert1, vert2, &tris6[i]);
	  }
	  else
	    i--;    
	}
      inters = mkIntersection_big(t,u,v,point);

       start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  ar_hits+=arenberg_other_bary_pre(&rays[i], &tris6[i], inters);
      ar_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + ari]=((float)ar_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  ar2_hits+=arenberg_other_bary_pre2(&rays[i], &tris6[i], inters);
      ar2_time += clock() - start_time;
      results_bary[reps*NR_ALGOS + ar2i]=((float)ar2_time)/cps;
      
      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  ar_hits+=arenberg_other_t_pre(&rays[i], &tris6[i], inters);
      ar_time += clock() - start_time;
      results_t[reps*NR_ALGOS + ari]=((float)ar_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  ar2_hits+=arenberg_other_t_pre2(&rays[i], &tris6[i], inters);
      ar2_time += clock() - start_time;
      results_t[reps*NR_ALGOS + ar2i]=((float)ar2_time)/cps;
      
      
      t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
      t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
      
      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  ar_hits+=arenberg_other_pre(&rays[i], &tris6[i], inters);
      ar_time += clock() - start_time;
      results[reps*NR_ALGOS + ari]=((float)ar_time)/cps;

      start_time = clock(); 
      for(k=0;k<OUTER_RUNS; k++)
	for(i=0;i<N;i++) for(l=0;l<INNER_RUNS;l++)
	  ar2_hits+=arenberg_other_pre2(&rays[i], &tris6[i], inters);
      ar2_time += clock() - start_time;
      results[reps*NR_ALGOS + ar2i]=((float)ar2_time)/cps;

      reps++;
      hits_so_far = 0;
      miss_so_far = 0;
 
      free(inters);
    }
         
    nr_hits += N/NR_SCALES;
    t0_hits=t1_hits=t2_hits=t3_hits=b0_hits=s0_hits=c0_hits=p0_hits=cr0_hits =pl0_hits=su0_hits=c1_hits=e0_hits=m0_hits=o0_hits=t01_hits=b1_hits=h0_hits=h1_hits=a122_hits=a2D_hits=sn_hits=occw_hits=ar_hits=cr3_hits=ar2_hits=h1_hits=0;
    t0_time=t1_time=t2_time=t3_time=b0_time=s0_time=c0_time=p0_time=cr0_time=pl0_time=su0_time=c1_time=e0_time=m0_time=o0_time=t01_time=b1_time=h0_time=h1_time=a122_time=a2D_time=sn_time=occw_time=ar_time=cr3_time=ar2_time=h1_time=0;
    
  }
  printf("\n");
  printf("\n");
  
  
  
  /*Printbaryresults*/

   num_sumresults(results_bary,start_h,stop_h);
  for(m=0;m<NR_ALGOS;m++)
    mkAlgoVal(results_bary[NR_ALGOS*(NR_SCALES+1)+m],m,&test_res[m]);
  
  qsort(test_res,NR_ALGOS,sizeof(algo_val),comp_val);
  printf("Fastest barycentric algorithms, at hitrates %.1f - %.1f: ", start_h, (stop_h));
  for(m=0;m<4;m++){
    printval(&test_res[m]);
    printf(", ");
  }
  printval(&test_res[m]);
  printf("\n");

  /*Print T-VALUE result*/
  
  num_sumresults(results_t,start_h,stop_h);
  for(m=0;m<NR_ALGOS;m++)
    mkAlgoVal(results_t[NR_ALGOS*(NR_SCALES+1)+m],m,&test_res[m]);
  
  qsort(test_res,NR_ALGOS,sizeof(algo_val),comp_val);
  printf("Fastest t-value algorithms, at hitrates %.1f - %.1f: ", start_h, (stop_h));
  for(m=0;m<4;m++){
    printval(&test_res[m]);
    printf(", ");
  }
  printval(&test_res[m]);
  printf("\n");
  
  /*print BOOL result*/

    
  num_sumresults(results,start_h,stop_h);
  for(m=0;m<NR_ALGOS;m++)
    mkAlgoVal(results[NR_ALGOS*(NR_SCALES+1)+m],m,&test_res[m]);
  
  qsort(test_res,NR_ALGOS,sizeof(algo_val),comp_val);
  
  printf("Fastest boolean algorithms, at hitrates %.1f - %.1f: ", start_h, (stop_h));
  for(m=0;m<4;m++){
    printval(&test_res[m]);
    printf(", ");
  }
  printval(&test_res[m]);
  printf("\n");
  
  return 0;
}
