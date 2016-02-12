/* advicer_constants.h*/


#define N 100000 /* size of testset i.e. number of ray triangle pairs*/
#define NR_SETS 1 /* could test on more testsets then one */
#define OUTER_RUNS 25 /* number of outer loops to run to have a measurable executiontime for every algorithms. 25 is enough for machines up to approximately 4GHz. In the future this figure should be increased with faster machines */

#define INNER_RUNS 1 /* this gives a possibility to test on each pair more times instead of running a lot of outer loos this will give a different behaivior cache-wise*/
#define NR_SCALES 10 /* this gives an increas of 0.1 between every hitrate*/
#define NR_PASS 5 
#define NR_ALGOS 22 /* The number of algorithms */
#define ALL (NR_SETS *OUTER_RUNS *INNER_RUNS)

#include "advicer_algorithms.h"

/* enumeration of all included algorithms */
enum algos {mt0,mt1,mt2,mt3,ma,pu,or,orc,mapl,pupl,ch1p,ch2p,ch3p,orp,orcp,hfp,hf2p,a2dp,hfh,hf2h,ari,ar2i};

typedef struct
{
  float value;
  int algo; 
}algo_val;
