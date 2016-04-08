//
//         Stephen Vincent and David Forsey.
//         Fast and accurate parametric curve length computation.
//         Journal of graphics tools, 6(4):29-40, 2001
//              
//

static double SurfaceArea
    ( double min_u ,
    double max_u ,
    int n_eval_pts_u ,
    double min_v ,
    double max_v ,
    int n_eval_pts_v )
{

    int i;
    double area = 0.0;
    double* u = NULL;
    double* v = NULL;
    TPoint3* pt = NULL;
    int u_idx;
    int v_idx;

    if ( !Odd ( n_eval_pts_u ) )
        n_eval_pts_u++;

    if ( !Odd ( n_eval_pts_v ) )
        n_eval_pts_v++;

    u = new double [ n_eval_pts_u ];
    v = new double [ n_eval_pts_v ];

    pt = new TPoint3 [ n_eval_pts_u * n_eval_pts_v ];

    // Evaluate points on the surface. Horner's rule could usefully be
    // used here if the surface can be represented as a polynomial.
    // Points are evaluated along u first, then v.
    // So an index of i represents a value in v_idx of i/n_eval_pts_u
    // and a value in u_idx of i%n_eval_pts_u

    i = 0;

    for ( v_idx = 0 ; v_idx < n_eval_pts_v ; ++v_idx )
    {

        v [ v_idx ] = min_v + ( max_v - min_v ) * (double)v_idx / ( n_eval_pts_v - 1 );

        for ( u_idx = 0 ; u_idx < n_eval_pts_u ; ++u_idx )
        {

            u [ u_idx ] = min_u + ( max_u - min_u ) * (double)u_idx / ( n_eval_pts_u - 1 );

            GetPoint ( u [ u_idx ] , v [ v_idx ] , &pt [ i ] );

            i++;

        }

    }

    // Compute areas of individual patch elements

    for ( v_idx = 0 ; v_idx < n_eval_pts_v - 1 ; v_idx += 2 )
    {

        i = v_idx * n_eval_pts_u;

        for ( u_idx = 0 ; u_idx < n_eval_pts_u - 1 ; u_idx += 2 )
        {

            area += GetSectionArea
                ( u [ u_idx ] ,
                u [ u_idx + 2 ] ,
                v [ v_idx ] ,
                v [ v_idx + 2 ] ,
                pt [ i ] ,
                pt [ i + 1 ] ,
                pt [ i + 2 ] ,
                pt [ n_eval_pts_u + i ] ,
                pt [ n_eval_pts_u + i + 1 ] ,
                pt [ n_eval_pts_u + i + 2 ] ,
                pt [ 2*n_eval_pts_u + i ] ,
                pt [ 2*n_eval_pts_u + i + 1 ] ,
                pt [ 2*n_eval_pts_u + i + 2 ] );

            i += 2;

        }

    }

    delete [] pt;
    delete [] v;
    delete [] u;

    return area;
}