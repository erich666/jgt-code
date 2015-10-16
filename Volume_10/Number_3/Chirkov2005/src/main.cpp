#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cTimer.h"
#include "idata.h"

#define M_PI 3.141592654

/* org jgt */
int intersect_triangle(const RAYTRI *rt);
int intersect_triangle1(const RAYTRI *rt);
int intersect_triangle1_(const RAYTRI *rt);
int intersect_triangle2(const RAYTRI *rt);
int intersect_triangle2_(const RAYTRI *rt);
int intersect_triangle3(const RAYTRI *rt);


int c2005_0(const RAYTRI *rt);
int c2005_2(const RAYTRI *rt);
int c2005_2_(const RAYTRI *rt);
int c2005_3(const RAYTRI *rt);
int c2005_3_(const RAYTRI *rt);


//statistics
int nMuls, nAdds, nDivs;
void stat(int a, int m, int d)
{
	nAdds += a;
	nMuls += m;
	nDivs += d;
}

#define RANDFLOAT (float(rand())/RAND_MAX)

int _cdecl main(int argc, char *argv[])
{
	srand(0);
	if(argc==4)
	{
		int fastest=0;
		int q;
		int result[10];
		int res;
		cTimer timer[10];
		RAYTRI *raytris;

		int num=atoi(argv[1]);
		int repeat=atoi(argv[2]);
		double percentHits = atof(argv[3]);


		raytris = (RAYTRI*)malloc(sizeof(RAYTRI)*num);
		if(!raytris)
		{
			printf("Err: could not alloc mem for raytris\n");
			exit(0);
		}

		/* set up ray-tri pairs */
		int nHits=0;
		for(q=0;q<num;q++)
		{
			raytris[q].v0[0] = RANDFLOAT;
			raytris[q].v0[1] = RANDFLOAT;
			raytris[q].v0[2] = RANDFLOAT;

			raytris[q].v1[0] = RANDFLOAT;
			raytris[q].v1[1] = RANDFLOAT;
			raytris[q].v1[2] = RANDFLOAT;

			raytris[q].v2[0] = RANDFLOAT;
			raytris[q].v2[1] = RANDFLOAT;
			raytris[q].v2[2] = RANDFLOAT;

			double ax = raytris[q].v1[0] - raytris[q].v0[0];
			double ay = raytris[q].v1[1] - raytris[q].v0[1];
			double az = raytris[q].v1[2] - raytris[q].v0[2];
			double bx = raytris[q].v2[0] - raytris[q].v0[0];
			double by = raytris[q].v2[1] - raytris[q].v0[1];
			double bz = raytris[q].v2[2] - raytris[q].v0[2];
			raytris[q].plane.x = ay * bz - az * by;
			raytris[q].plane.y = az * bx - ax * bz;
			raytris[q].plane.z = ax * by - ay * bx;
			raytris[q].plane.d = 
				raytris[q].plane.x*raytris[q].v0[0] + 
				raytris[q].plane.y*raytris[q].v0[1] + 
				raytris[q].plane.z*raytris[q].v0[2];

			static const double SQ = sqrt(1.0/3.0);
			double len = (raytris[q].plane.x*raytris[q].plane.x + 
				raytris[q].plane.y*raytris[q].plane.y + 
				raytris[q].plane.z*raytris[q].plane.z)*SQ;

			if(fabs(raytris[q].plane.x)>len)
				raytris[q].plane.type=RAYTRI::PLANE::X;
			else
				if(fabs(raytris[q].plane.y)>len)
					raytris[q].plane.type=RAYTRI::PLANE::Y;
				else
					raytris[q].plane.type=RAYTRI::PLANE::Z;

				raytris[q].org[0] = RANDFLOAT;
				raytris[q].org[1] = RANDFLOAT;
				raytris[q].org[2] = RANDFLOAT;

				raytris[q].end[0] = RANDFLOAT;
				raytris[q].end[1] = RANDFLOAT;
				raytris[q].end[2] = RANDFLOAT;

				raytris[q].dir[0] = raytris[q].end[0] - raytris[q].org[0];
				raytris[q].dir[1] = raytris[q].end[1] - raytris[q].org[1];
				raytris[q].dir[2] = raytris[q].end[2] - raytris[q].org[2];

				if(percentHits>=0.0)
				{
					bool lh = nHits>int(num*percentHits);
					bool hit = c2005_3(&raytris[q])==0;
					if(lh^hit)
					{
						q--;
						continue;
					}
					nHits++;
				}

		}

		for(q=0;q<10;q++)
		{
			result[q]=0;
			timer[q].reset();
		}

		res=0;
		timer[0].start();
		for(int w=0;w<repeat;w++)
		{
			for(q=0;q<num;q++)
			{
				res=intersect_triangle(&raytris[q]);
				result[0]+=res;
			}
		}
		timer[0].stop();
		timer[0].multByFactor(1.0f/(repeat*num));
		result[0]/=repeat;

		res=0;
		timer[1].start();
		for(w=0;w<repeat;w++)
		{
			for(q=0;q<num;q++)
			{
				res=intersect_triangle1(&raytris[q]);
				result[1]+=res;
			}
		}
		timer[1].stop();
		timer[1].multByFactor(1.0f/(repeat*num));
		result[1]/=repeat;


		res=0;
		timer[2].start();
		for(w=0;w<repeat;w++)
		{
			for(q=0;q<num;q++)
			{
				res=intersect_triangle2(&raytris[q]);
				result[2]+=res;
			}
		}
		timer[2].stop();
		timer[2].multByFactor(1.0f/(repeat*num));
		result[2]/=repeat;


		res=0;
		timer[3].start();
		for(w=0;w<repeat;w++)
		{
			for(q=0;q<num;q++)
			{
				res=intersect_triangle3(&raytris[q]);
				result[3]+=res;
			}
		}
		timer[3].stop();
		timer[3].multByFactor(1.0f/(repeat*num));
		result[3]/=repeat;


		res=0;
		timer[4].start();
		for(w=0;w<repeat;w++)
		{
			for(q=0;q<num;q++)
			{
				res=c2005_0(&raytris[q]);
				result[4]+=res;
			}
		}
		timer[4].stop();
		timer[4].multByFactor(1.0f/(repeat*num));
		result[4]/=repeat;

		res=0;
		timer[5].start();
		for(w=0;w<repeat;w++)
		{
			for(q=0;q<num;q++)
			{
				res=c2005_2(&raytris[q]);
				result[5]+=res;
			}
		}
		timer[5].stop();
		timer[5].multByFactor(1.0f/(repeat*num));
		result[5]/=repeat;


		res=0;
		timer[6].start();
		for(w=0;w<repeat;w++)
		{
			for(q=0;q<num;q++)
			{
				res=c2005_3(&raytris[q]);
				result[6]+=res;
			}
		}
		timer[6].stop();
		timer[6].multByFactor(1.0f/(repeat*num));
		result[6]/=repeat;


		for(q=0;q<8;q++) if(timer[q].getTime()<timer[fastest].getTime()) fastest=q;

		printf("-----------------------------------------------------------\n");
		printf("%s 0: %d (%d): time=%2.2f ns (original jgt code)        \n",fastest==0 ? "**":"  ",result[0],num,timer[0].getTime()*1.0e9);
		printf("%s 1: %d (%d): time=%2.2f ns (divide at end)            \n",fastest==1 ? "**":"  ",result[1],num,timer[1].getTime()*1.0e9);
		printf("%s 2: %d (%d): time=%2.2f ns (div early)                \n",fastest==2 ? "**":"  ",result[2],num,timer[2].getTime()*1.0e9);
		printf("%s 3: %d (%d): time=%2.2f ns (div early+cross before if)\n",fastest==3 ? "**":"  ",result[3],num,timer[3].getTime()*1.0e9);
		printf("%s 4: %d (%d): time=%2.2f ns (c2005_0)                 \n",fastest==4 ? "**":"  ",result[4],num,timer[4].getTime()*1.0e9);
		printf("%s 5: %d (%d): time=%2.2f ns (c2005_2)                 \n",fastest==5 ? "**":"  ",result[5],num,timer[5].getTime()*1.0e9);
		printf("%s 6: %d (%d): time=%2.2f ns (c2005_3)                 \n",fastest==6 ? "**":"  ",result[6],num,timer[6].getTime()*1.0e9);
		int mFast=0;
		double mTime = 1e10;
		for(int f=0; f<4; f++)
			if(timer[f].getTime()<mTime)
			{
				mTime = timer[f].getTime();
				mFast = f;
			}

		int cFast=4;
		double cTime = 1e10;
		for(int f=4; f<7; f++)
			if(timer[f].getTime()<cTime)
			{
				cTime = timer[f].getTime();
				cFast = f;
			}
		printf("Algo %d is %2.2f %% faster than algo %d\n",cFast,100.0*((float)timer[mFast].getTime()/timer[cFast].getTime()-1.0), mFast);



		printf("\n");

		nMuls = nAdds = nDivs = 0;
		res=0;
		for(q=0;q<num;q++)
			res += intersect_triangle1_(&raytris[q]);
		printf("divide at end: %d (%d), muls: %2.2f, adds: %2.2f, divs: %2.2f\n", res, num, nMuls/float(num), nAdds/float(num), nDivs/float(num));

		nMuls = nAdds = nDivs = 0;
		res=0;
		for(q=0;q<num;q++)
			res += intersect_triangle2_(&raytris[q]);
		printf("div early    : %d (%d), muls: %2.2f, adds: %2.2f, divs: %2.2f\n", res, num, nMuls/float(num), nAdds/float(num), nDivs/float(num));

		nMuls = nAdds = nDivs = 0;
		res=0;
		for(q=0;q<num;q++)
			res += c2005_2_(&raytris[q]);
		printf("c2005_2     : %d (%d), muls: %2.2f, adds: %2.2f, divs: %2.2f\n", res, num, nMuls/float(num), nAdds/float(num), nDivs/float(num));

		nMuls = nAdds = nDivs = 0;
		res=0;
		for(q=0;q<num;q++)
			res += c2005_3_(&raytris[q]);
		printf("c2005_3     : %d (%d), muls: %2.2f, adds: %2.2f, divs: %2.2f\n", res, num, nMuls/float(num), nAdds/float(num), nDivs/float(num));

	}
	else printf("Usage: %s numTriangles numTests hitPercentage\n",argv[0]);

	return 0;
}

