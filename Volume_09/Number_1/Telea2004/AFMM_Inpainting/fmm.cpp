#include "fmm.h"
#include "flags.h"
#include <math.h>
#include <iostream.h>
#include <signal.h>

struct 	NewValue {  int i; int j; float value;  };	//Used in the diffuse() routine





FastMarchingMethod::FastMarchingMethod(FIELD<float>* f_,FLAGS* flags_,int N_)
		   :f(f_),flags(flags_),ptrs(f_->dimX(),f_->dimY()),N(N_)
{
   for(int j=0;j<flags->dimY();j++)
      for(int i=0;i<flags->dimX();i++)
	 if (flags->narrowband(i,j))
	 {
	    std::multimap<float,Coord>::value_type v(f->value(i,j),Coord(i,j));
	    ptrs.value(i,j) = map.insert(v);
         } 
}


FastMarchingMethod::~FastMarchingMethod()
{  }


int FastMarchingMethod::execute(int& negd_, int& nextr_, float maxf_)
{
   int cc;

   negd = 0; nextr = 0; maxf = maxf_;
   for(iteration=0,cc=0;iteration<N;iteration++,cc++)
   {
      if (!diffuse()) break;

      if (cc==1000)
      {
	cout<<"Iteration "<<iteration<<" done"<<endl;
        cc=0;
      }
   }

   negd_ = negd; nextr_ = nextr;                //return stats to caller
   return iteration;
}




void FastMarchingMethod::solve(int fi_1j,int fij_1,float vi_1j,float vij_1,float& sol)
{
	float ss,d,r;

	if (fi_1j == FLAGS::ALIVE || fi_1j == FLAGS::EXTREMUM)
	  if (fij_1 == FLAGS::ALIVE || fij_1 == FLAGS::EXTREMUM)	//sol determined by two points: solve order-2 equation
	  {
	     d   = 2 - SQR(vi_1j-vij_1);
	     if (d>=0)
	     {	
	       r   = sqrt(d);
	       ss  = ((vi_1j+vij_1) - r)/2; 
	       if (ss >= vi_1j && ss >= vij_1) sol = MIN(sol,ss);
	       else
	       {
	          ss += r;
	          if (ss >= vi_1j && ss >= vij_1) sol = MIN(sol,ss);
	       }
	     }
	     else { negd++; return; }					//should never happen, but still...
          }
	  else sol = MIN(sol,1 + vi_1j); 				//sol determined by one point: solve order-1 equation
        else        
          if (fij_1 == FLAGS::ALIVE || fij_1 == FLAGS::EXTREMUM) 
             sol = MIN(sol,1 + vij_1); 					//sol determined by one point: solve order-1 equation
}



int FastMarchingMethod::diffuse()
{
    static NewValue newp[20]; 

    //*** 1. FIND POINT IN NARROWBAND WITH LOWEST DISTANCE-VALUE
    int min_i,min_j;
    std::multimap<float,Coord>::iterator it=map.begin();
    if (it==map.end()) return 0;

    min_i  = (*it).second.i;
    min_j  = (*it).second.j;
    map.erase(it);					//erase point from 'map', since we'll make it alive in step 2

    //*** 2. MAKE MIN-POINT ALIVE
    flags->value(min_i,min_j) = FLAGS::ALIVE;		
    if (f->value(min_i,min_j)>=maxf) return 1;		//stop evolution if we reached the user-prescribed threshold.
   


     //*** 3. FIND ALL ITS STILL-TO-BE-UPDATED NEIGHBOURS
    Coord nbs[4]; int nn = 0;
    tag_nbs(min_i-1,min_j,min_i,min_j,nbs,nn);
    tag_nbs(min_i+1,min_j,min_i,min_j,nbs,nn);
    tag_nbs(min_i,min_j-1,min_i,min_j,nbs,nn);
    tag_nbs(min_i,min_j+1,min_i,min_j,nbs,nn);

    if (!nn) 						//only alive-neighbours of point (min_i,min_j) found,
    { 							//so it should be an extremum point...
	flags->value(min_i,min_j) = FLAGS::EXTREMUM;
        nextr++;
	return 1;	
    }

    //*** 4. UPDATE VALUES OF NEIGHBOURS OF MIN-POINT
    NewValue* nnewp = newp;				//start updating neighbours. Their new values will be saved
    for(nn--;nn>=0;nn--)				//in nnewp[] and pasted back in 'f' at the update end.
    {
	int i = nbs[nn].i;
	int j = nbs[nn].j;

	float vi_1j = f->value(i-1,j),     vijx1 = f->value(i,j+1);
	float vix1j = f->value(i+1,j),     vij_1 = f->value(i,j-1);
	int   fi_1j = flags->value(i-1,j), fijx1 = flags->value(i,j+1);
	int   fix1j = flags->value(i+1,j), fij_1 = flags->value(i,j-1);

	float sol = INFINITY;
	solve(fi_1j,fij_1,vi_1j,vij_1,sol);
	solve(fix1j,fij_1,vix1j,vij_1,sol);
	solve(fi_1j,fijx1,vi_1j,vijx1,sol); 
	solve(fix1j,fijx1,vix1j,vijx1,sol); 

	if (sol < INFINITY/2) 
        { nnewp->i = i; nnewp->j = j; nnewp->value = sol; nnewp++; } 
    }

    //***5. Write updated values back in field.
    for(nnewp--;nnewp>=newp;nnewp--)				//for all updated neighbours:
    {
       map.erase(ptrs.value(nnewp->i,nnewp->j));		//remove the neighbour's entry from the sorted map...
       std::multimap<float,Coord>::value_type v(nnewp->value,Coord(nnewp->i,nnewp->j));
       ptrs.value(nnewp->i,nnewp->j) = map.insert(v);		//...and insert it back since its field-value changed		
       f->value(nnewp->i,nnewp->j) = nnewp->value;		//update the field too!
   }
 
    return 1;
}







void FastMarchingMethod::tag_nbs(int min_i,int min_j,		//Gathers all neighbours (min_i,min_j) of (i,j)
                                 int i,int j,Coord* nbs,	//are to be updated because updating (i,j).
				 int& nn)
{
    if (min_i>=0 && min_i<flags->dimX() && min_j>=0 && min_j<flags->dimY() && 
        !flags->alive(min_i,min_j) && !flags->extremum(min_i,min_j))
    {
       nbs[nn].i = min_i; nbs[nn].j = min_j; nn++;
       if (flags->value(min_i,min_j)!=FLAGS::NARROW_BAND)
          add_to_narrowband(min_i,min_j,i,j);			//Here we do our gathering stuff!
    }
}



void FastMarchingMethod::add_to_narrowband(int i,int j,int,int)	//Adds point i,j to narrowband.
{
          flags->value(i,j) = FLAGS::NARROW_BAND;
	  std::multimap<float,Coord>::value_type v(f->value(i,j),Coord(i,j));
 	  ptrs.value(i,j)  = map.insert(v);
}



