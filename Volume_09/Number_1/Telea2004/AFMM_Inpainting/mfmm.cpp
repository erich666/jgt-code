#include "mfmm.h"
#include "flags.h"
#include "dqueue.h"
#include "stack.h"
#include <math.h>
#include <iostream.h>


struct  NewValue {  int i; int j; float value;  }; 


ModifiedFastMarchingMethod::ModifiedFastMarchingMethod
			    (FIELD<float>* f_,FLAGS* flags_,IMAGE<float>* image_,
			     FIELD<float>* gx,FIELD<float>* gy,FIELD<float>* d,int br,
			     int dst_wt,int lev_wt,int N_)
		   :FastMarchingMethod(f_,flags_,N_),image(image_),grad_x(gx),grad_y(gy),dist(d),
		    B_radius(br),dst_weighting(dst_wt),lev_weighting(lev_wt)
{
}




int ModifiedFastMarchingMethod::execute(int& negd_, int& nextr_, float maxf_)
{
   int ret = FastMarchingMethod::execute(negd_,nextr_,maxf_);
 							//2. Call inherited fast-marching-method that will
							//   do all the evolution job...
   return ret;
}



void ModifiedFastMarchingMethod::add_to_narrowband(int i,int j,int active_i,int active_j)
{
    float gx_r=0,gy_r=0,gx_g=0,gy_g=0,gx_b=0,gy_b=0,im_r=0,im_g=0,im_b=0; int ii,jj;
    float cnt=0,cntx=0,cnty=0,r;

    float dst0 = dist->value(i,j);
    int N = B_radius;

    for(ii=-N;ii<=N;ii++)					//look at the known pixels in a window around current-point
     for(jj=-N;jj<=N;jj++)
     {
       if (!flags->alive(i+ii,j+jj)) continue;			//work on known pixels only...
       if (ii==0 || jj==0) continue;				//skip current point, we inpaint it
       float dirx = -ii; float diry = -jj;			//project direction (i,j)->current-point on image gradient
       float dd  = sqrt(dirx*dirx+diry*diry);
       if (dd>N) continue;
       float ndirx = dirx/dd, ndiry = diry/dd;			//(ndirx,ndiry) is unit-direction (i,j)->(i+ii,j+jj)
       float dst = dist->value(i+ii,j+jj);
       r = ndirx*grad_x->value(i,j) + ndiry*grad_y->value(i,j);	//do directional weighting
       r = fabs(r);
       if (dst_weighting) r /= dd*dd;				//do distance weighting (optional)
       if (lev_weighting) r /= (1+(dst-dst0)*(dst-dst0));	//do level-weighting (optional)

       if (!flags->faraway(i+ii+1,j+jj) && !flags->faraway(i+ii-1,j+jj) &&
           !flags->faraway(i+ii,j+jj+1) && !flags->faraway(i+ii,j+jj-1))
       {
           float igx = image->r.value(i+ii+1,j+jj)-image->r.value(i+ii-1,j+jj);
	   float igy = image->r.value(i+ii,j+jj+1)-image->r.value(i+ii,j+jj-1);
	   float  il = sqrt(igx*igx+igy*igy);
	   float f   = il + igy * grad_x->value(i+ii,j+jj) - igx * grad_y->value(i+ii,j+jj);
	   //r *= 1/(1+f);
       }

       //r = r*fabs(grad_x->value(i+ii,j+jj)*grad_x->value(i,j)+grad_y->value(i+ii,j+jj)*grad_y->value(i,j));

       im_r += r*image->r.value(i+ii,j+jj);			//computed image-avg weighted by the above projection
       im_g += r*image->g.value(i+ii,j+jj);			//as well as image-gradient weighted by above projection
       im_b += r*image->b.value(i+ii,j+jj);
       cnt+=r;

       if (!flags->faraway(i+ii+1,j+jj) && !flags->faraway(i+ii-1,j+jj))
       {
	  gx_r += dirx*r*(image->r.value(i+ii+1,j+jj)-image->r.value(i+ii-1,j+jj)); 
	  gx_g += dirx*r*(image->g.value(i+ii+1,j+jj)-image->g.value(i+ii-1,j+jj)); 
	  gx_b += dirx*r*(image->b.value(i+ii+1,j+jj)-image->b.value(i+ii-1,j+jj)); 



	  cntx += r;
       }
       if (!flags->faraway(i+ii,j+jj+1) && !flags->faraway(i+ii,j+jj-1))
       {
          gy_r += diry*r*(image->r.value(i+ii,j+jj+1)-image->r.value(i+ii,j+jj-1));
	  gy_g += diry*r*(image->g.value(i+ii,j+jj+1)-image->g.value(i+ii,j+jj-1));
	  gy_b += diry*r*(image->b.value(i+ii,j+jj+1)-image->b.value(i+ii,j+jj-1));
	  cnty += r;
       }
     }

     float c_r=0,c_g=0,c_b=0;
     if (cnt==0 || cntx==0 || cnty==0)                            //occurs sometimes when B_radius very small (e.g. 1)
     {
       im_r=im_g=im_b=0; cnt=0;   
       for(ii=i-2;ii<=i+2;ii++)
         for(jj=j-2;jj<=j+2;jj++)
            if (!flags->faraway(ii,jj)) 
	    { im_r += image->r.value(ii,jj); im_g += image->g.value(ii,jj); im_b += image->b.value(ii,jj); cnt++; }
     }
     else
     {
       gx_r /= cntx; gy_r /= cnty;                                //normalize avg-gradient
       gx_g /= cntx; gy_g /= cnty;
       gx_b /= cntx; gy_b /= cnty;
       r = sqrt(gx_r*gx_r+gy_r*gy_r); if (r>0.00001) {gx_r/=r; gy_r/= r;}
       r = sqrt(gx_g*gx_g+gy_g*gy_g); if (r>0.00001) {gx_g/=r; gy_g/= r;}
       r = sqrt(gx_b*gx_b+gy_b*gy_b); if (r>0.00001) {gx_b/=r; gy_b/= r;}
       
       c_r = gx_r + gy_r;
       c_g = gx_g + gy_g;
       c_b = gx_b + gy_b;
     }
     
		                                                  //im = avg-perception of image-neighorhood
     image->r.value(i,j) = im_r/cnt + c_r;			  //c  = avg-gradient of image in direction of gradient of DT
     image->g.value(i,j) = im_g/cnt + c_g;
     image->b.value(i,j) = im_b/cnt + c_b;

     FastMarchingMethod::add_to_narrowband(i,j,active_i,active_j);
}


/*
void ModifiedFastMarchingMethod::add_to_narrowband(int i,int j,int active_i,int active_j)
{
    float gx_r=0,gy_r=0,gx_g=0,gy_g=0,gx_b=0,gy_b=0,im_r=0,im_g=0,im_b=0; int ii,jj;
    float cnt=0,cntx=0,cnty=0,r;

    float dst0 = dist->value(i,j);
    float cx=0,cy=0;
    int N = B_radius;

    for(ii=-N;ii<=N;ii++)				//look at the known pixels in a window around current-point
     for(jj=-N;jj<=N;jj++)
     {
       if (!flags->alive(i+ii,j+jj)) continue;			//work on known pixels only...
       if (ii==0 || jj==0) continue;				//skip current point, we inpaint it
       float dirx = -ii; float diry = -jj;			//project direction (i,j)->current-point on image gradient
       float dd  = sqrt(dirx*dirx+diry*diry);
       if (dd>N) continue;
       float dst = dist->value(i+ii,j+jj);
       dirx /= dd; diry /= dd; 
       r = dirx*grad_x->value(i,j) + diry*grad_y->value(i,j);	//do directional weighting
       r = fabs(r);
       if (dst_weighting) r /= dd*dd;				//do distance weighting (optional)
       if (lev_weighting) r /= (1+(dst-dst0)*(dst-dst0));	//do level-weighting (optional)

       //r = r*fabs(grad_x->value(i+ii,j+jj)*grad_x->value(i,j)+grad_y->value(i+ii,j+jj)*grad_y->value(i,j));

       im_r += r*image->r.value(i+ii,j+jj);			//computed image-avg weighted by the above projection
       im_g += r*image->g.value(i+ii,j+jj);			//as well as image-gradient weighted by above projection
       im_b += r*image->b.value(i+ii,j+jj);
       cnt+=r;

       cx += r*(i+ii); cy += r*(j+jj);

       if (!flags->faraway(i+ii+1,j+jj) && !flags->faraway(i+ii-1,j+jj))
       {
	  gx_r += r*(image->r.value(i+ii+1,j+jj)-image->r.value(i+ii-1,j+jj)); 
	  gx_g += r*(image->g.value(i+ii+1,j+jj)-image->g.value(i+ii-1,j+jj)); 
	  gx_b += r*(image->b.value(i+ii+1,j+jj)-image->b.value(i+ii-1,j+jj)); 
	  cntx += r;
       }
       if (!flags->faraway(i+ii,j+jj+1) && !flags->faraway(i+ii,j+jj-1))
       {
          gy_r += r*(image->r.value(i+ii,j+jj+1)-image->r.value(i+ii,j+jj-1));
	  gy_g += r*(image->g.value(i+ii,j+jj+1)-image->g.value(i+ii,j+jj-1));
	  gy_b += r*(image->b.value(i+ii,j+jj+1)-image->b.value(i+ii,j+jj-1));
	  cnty += r;
       }
     }

     float c_r=0,c_g=0,c_b=0;
     if (cnt==0 || cntx==0 || cnty==0)                            //occurs sometimes when B_radius very small (e.g. 1)
     {
       im_r=im_g=im_b=0; cnt=0;   
       for(ii=i-2;ii<=i+2;ii++)
         for(jj=j-2;jj<=j+2;jj++)
            if (!flags->faraway(ii,jj)) 
	    { im_r += image->r.value(ii,jj); im_g += image->g.value(ii,jj); im_b += image->b.value(ii,jj); cnt++; }
     }
     else
     {
       float dx = grad_x->value(i,j), dy = grad_y->value(i,j);    //get distance-func gradient
       gx_r /= cntx; gy_r /= cnty;                                //normalize avg-gradient
       gx_g /= cntx; gy_g /= cnty;
       gx_b /= cntx; gy_b /= cnty;
       r = sqrt(gx_r*gx_r+gy_r*gy_r); if (r>0.00001) {gx_r/=r; gy_r/= r;}
       r = sqrt(gx_g*gx_g+gy_g*gy_g); if (r>0.00001) {gx_g/=r; gy_g/= r;}
       r = sqrt(gx_b*gx_b+gy_b*gy_b); if (r>0.00001) {gx_b/=r; gy_b/= r;}
       c_r = dx*gx_r + dy*gy_r;                                  //dot product img-avg-grad * distfunc-grad
       c_g = dx*gx_g + dy*gy_g;
       c_b = dx*gx_b + dy*gy_b;
       
       cx /= cnt; cy /= cnt;
       float d = sqrt((cx-i)*(cx-i)+(cy-j)*(cy-j));
       c_r *= d; c_g *= d; c_b *= d;
       //!!!c *= sqrt((cx-i)*(cx-i)+(cy-j)*(cy-j));
     }
     
		                                                  //im = avg-perception of image-neighorhood
     image->r.value(i,j) = im_r/cnt + c_r;			  //c  = avg-gradient of image in direction of gradient of DT
     image->g.value(i,j) = im_g/cnt + c_g;
     image->b.value(i,j) = im_b/cnt + c_b;

     FastMarchingMethod::add_to_narrowband(i,j,active_i,active_j);
}
*/


int ModifiedFastMarchingMethod::diffuse()
{
    static NewValue newp[20]; 

    //*** 1. FIND MIN-POINT IN NARROWBAND WITH LOWEST VALUE
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

    if (!nn) 						//no more alive-neighbous of point (min_i,min_j) found,
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


