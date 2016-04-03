//
//         Stephen Vincent and David Forsey.
//         Fast and accurate parametric curve length computation.
//         Journal of graphics tools, 6(4):29-40, 2001
//              
//
//	Code to calculate the arc length of a parametric curve.
//	There are 2 routines that can be called , GetCurveLength and GetCurveLengthSlidingWindow.
//	The former is the basic algorithm : the latter includes the sliding window modification.
//	Both routines require an estimate of how many points should initially be calculated along
//	the curve : as discussed in the text a value of 31 will give an accuracy of a few parts in
//	a million for a cubic Bezier curve evaluated from 0 to 1

//	The user must supply the routine GetPoint

#include <stdio.h>
#include <math.h>

#define	Sqr(x)			((x) *(x))
#define	Abs(x)			((x) > 0 ? (x) : -(x))
#define	Odd(x)			((x) & 0x01)


typedef struct 
{
	double	x;
	double	y;
	double	z;
} TPoint3;

//	A routine to evaluate the point on the curve at position t

extern void GetPoint ( double t , TPoint3*	pt );

double Distance ( TPoint3*	pt_a , TPoint3*	pt_b )

{
	return ( sqrt ( Sqr ( pt_a->x - pt_b->x ) +
			 		Sqr ( pt_a->y - pt_b->y ) +
			  		Sqr ( pt_a->z - pt_b->z ) ) );
}

static double GetSectionLength
					( double		t0 ,
					  double		t1 ,
					  double		t2 ,
					  TPoint3		pt0 ,
					  TPoint3		pt1 ,
					  TPoint3		pt2 )

//	Compute the length of a small section of a parametric curve from
//	t0 to t2 , recursing if necessary. t1 is the mid-point.
//	The 3 points at these parametric values are precomputed.

{

	const double	kEpsilon		= 1e-5;
	const double	kEpsilon2		= 1e-6;
	const double	kMaxArc			= 1.05;
	const double	kLenRatio		= 1.2;
	
	double		d1 , d2;
	double		len_1 , len_2;
	double		da , db;
	
	d1 = Distance ( &pt0 , &pt2 );
	
	da = Distance ( &pt0 , &pt1 );
	db = Distance ( &pt1 , &pt2 );
	
	d2 = da + db;
	
	if ( d2 < kEpsilon )
	{
	
		return ( d2 + ( d2 - d1 ) / 3 );
			 
	}
	else if ( ( d1 < kEpsilon || d2/d1 > kMaxArc ) ||
			  ( da < kEpsilon2 || db/da > kLenRatio ) ||
			  ( db < kEpsilon2 || da/db > kLenRatio ) )
	{
	
		//	We're in a region of high curvature. Recurse.

		//	Lengths are tested against kEpsilon and kEpsilon2 just
		//	to prevent divison-by-zero/overflow.

		//	However kEpsilon2 should be less than half of kEpsilon 
		//	otherwise we'll get unnecessary recursion.

		//	The value of kMaxArc implicitly refers to the maximum
		//	angle that can be subtended by the circular arc that 
		//	approximates the curve between t and prev_t.
		//	The relationship is : kMaxArc = ( 1 - ø*ø/24 ) /
		//	( 1 - ø*ø/6 ).
		//	Rearranging gives ø = sqrt ( 24 * ( kMaxArc - 1 ) / 
		//	( 4 * kMaxArc - 1 ) )

		//	kLenRatio : when the lengths of da and db become too
		//	dissimilar the curve probably ( not necessarily ) 
		//	can't be approximated by a circular arc here.
		//	Recurse again : a value of 1.1 is a little high in 
		//	that it won't accurately detect a certain pathological 
		//	case of cusp mentioned in the documentation : on the 
		//	other hand too low a value results in unnecessary
		//	recursion.

		TPoint3		pt_mid;
		double		mid_t = ( t0 + t1 ) / 2;

		GetPoint ( mid_t , &pt_mid );

		len_1 = GetSectionLength
						( t0 ,
						  mid_t ,
						  t1 ,
						  pt0 ,
						  pt_mid ,
						  pt1 );						 
	
		mid_t = ( t1 + t2 ) / 2;
		GetPoint ( mid_t , &pt_mid );

		len_2 = GetSectionLength 
					   ( t1 ,
						 mid_t ,
						 t2 ,
						 pt1 ,
						 pt_mid ,
						 pt2 );						 
	
		return	( len_1 + len_2 );

	}
	else
	{
		return ( d2 + ( d2 - d1 ) / 3 );	 
	}
	
}

double GetCurveLength ( double	min_t ,
						double	max_t ,
						int		n_eval_pts )

//	Calculates the length of a parametric curve from min_t to max_t.

//	n_eval_pts points along the curve will be determined ( not 
//	allowing for any recursion that may be necessary )

{
	
	int			i;
	double		len = 0.0;
	double		*t;
	TPoint3		*pt;

	if ( !Odd ( n_eval_pts ) )
	  n_eval_pts++;
	  
	t = new double [ n_eval_pts ];
	pt = new TPoint3 [ n_eval_pts ];
	
	for ( i = 0 ; i < n_eval_pts ; ++i )
	{
	
		t [i]  = min_t + ( max_t - min_t ) * (double)i / ( n_eval_pts - 1 );

		GetPoint ( t [i] , &pt [i] );
	
	}	
	
	for ( i = 0 ; i < n_eval_pts - 1 ; i += 2 )
	{
	
		len += GetSectionLength ( t [i] , t [i+1] , t [i+2] ,
								  pt [i] , pt [i+1] , pt [i+2] );	
	}
	
	delete [] pt;
	delete [] t;
	
	return len;
	
}

double GetCurveLengthSlidingWindow
					( double	min_t ,
					  double	max_t ,
					  int		n_eval_pts )

//	Calculates the length of a parametric curve from min_t to max_t.

//	Coordinates on the curve are initially calculated at n_eval_pts
//	points ( n_eval_pts is required to be odd ) : more points may be
//	calculated at regions of high curvature.

//	The curve is divided into n_eval_pts - 1 segments. We get 2 
//	estimates for the length of each segment and average them
//	( except at the ends ). If the estimates differ by more than a
//	certain ratio recursively decompose that section to get a more
//	estimate.

	
{
	const double	kRatioError	= 2e-5;
	
	int			i;
	double		len = 0.0;
	double		len_a;
	double		len_b;
	double*		t = NULL;
	TPoint3*	pt = NULL;
	double		d1;
	double		d2;
	int			n_segs;
	double		t_mid;
	TPoint3		pt_mid;

	
	double		*seg_len_a = NULL;	
	double		*seg_len_b = NULL;
	
	double		*pt_distance = NULL;	//	Distances between successive points
	
	if ( !Odd ( n_eval_pts ) )
	  n_eval_pts++;
	
	n_segs = n_eval_pts - 1;
	
	pt = new TPoint3 [ n_eval_pts ];
	t = new double [ n_eval_pts ];
	
	seg_len_a = new double [ n_segs ];
	seg_len_b = new double [ n_segs ];
	pt_distance = new double [ n_segs ];
	
	//	Evaluate points along the curve. Horner's rule could usefully be
	//	used here if the curve can be represented as a polynomial.
	
	for ( i = 0 ; i < n_eval_pts ; ++i )
	{
	
		t [ i ] = min_t + ( max_t - min_t ) * (double)i / ( n_eval_pts - 1 );	
	
		GetPoint ( t [ i ] , &pt [ i ] );
		
	}
	
	//	Compute distances between successive points
	
	for ( i = 0 ; i < n_segs ; i++ )
	{
		pt_distance [ i ] = Distance ( &pt [i] , &pt [i+1] ); 
	}
	
	//	Get first estimate
	
	for ( i = 0 ; i < n_segs ; i += 2 )
	{

		d1 = Distance ( &pt [i] , &pt [i+2] );
		
		d2 = pt_distance [ i ] + pt_distance [ i+1 ];

		len_a = d2 + ( d2 - d1 ) / 3;	

		seg_len_a [ i ] = len_a * pt_distance [ i ] / d2;
		seg_len_a [ i+1 ] = len_a * pt_distance [ i+1 ] / d2;

	}
	
	//	Get second estimate of segment length for all segments
	//	except the first and last.
	
	for ( i = 1 ; i < n_segs - 1 ; i += 2 )
	{

		d1 = Distance ( &pt [i] , &pt [i+2] );
		
		d2 = pt_distance [ i ] + pt_distance [ i+1 ];

		len_b = d2 + ( d2 - d1 ) / 3;

		seg_len_b [ i ] = len_b * pt_distance [ i ] / d2;
		seg_len_b [ i+1 ] = len_b * pt_distance [ i+1 ] / d2;

	}
	
	for ( i = 0 ; i < n_segs ; i++ )
	{

		if ( i == 0 || i == n_segs - 1 )
		{
	
			//	First and last segments : call GetSectionLength to get an
			//	accurate estimate.
	
			t_mid = ( t [i] + t [i+1] ) / 2;
			GetPoint ( t_mid , &pt_mid );
	
			len += GetSectionLength ( t [i] , t_mid , t [i+1] , pt [i] , pt_mid , pt[i+1] );
		}
		else
		{

			if ( seg_len_b [ i ] < 1e-10 )
			{
				len += seg_len_a [ i ] / 2;
			}
			else
			{

				double ratio = seg_len_a [ i ] / seg_len_b [ i ];
		
				if ( ratio > 1 + kRatioError || ratio < 1 - kRatioError )
				{
					t_mid = ( t [i] + t [i+1] ) / 2;
					GetPoint ( t_mid , &pt_mid );
	
					len += GetSectionLength ( t [i] , t_mid , t [i+1] , pt [i] , pt_mid , pt[i+1] );
				}
				else
				{
					len += ( seg_len_a [ i ] + seg_len_b [ i ] ) / 2;
				}
	
			}
		}

	}
	
	delete [] pt_distance;
	delete [] seg_len_b;
	delete [] seg_len_a;
	delete [] t;
	delete [] pt;
	
	return len;
	
}
