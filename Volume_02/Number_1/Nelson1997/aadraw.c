/*
 * aadraw.c
 *
 *	The code that actually draws the lines.
 *
 *	The jaggy lines are not bug free.  They hit a few stray
 *	pixels.  Sorry.
 */

#include <math.h>
#include "aaline.h"



/* Tables that need to be initialized */
long int slope_corr_table[SC_TABLE_SIZE];
long int filter_table[F_TABLE_SIZE];
long int sqrt_table[SR_TABLE_SIZE];

/* Pointers to the frame buffer storage area */
long int *frame_buffer;
long int *z_buffer;



/*
 *---------------------------------------------------------------
 *
 * setup_line
 *
 *	Perform the setup operation for a line, then draw it
 *
 *---------------------------------------------------------------
 */

void
setup_line(aa_vertex *v1, aa_vertex *v2)
{
    float dx, dy;			/* Deltas in X and Y */
    float udx, udy;			/* Positive version of deltas */
    float dr, dg, db, da;		/* Deltas for RGBA */
    float one_du;			/* 1.0 / udx or udy */
    aa_setup_line line;

    dx = v1->x - v2->x;
    dy = v1->y - v2->y;
    if (dx < 0.0)
	udx = -dx;
    else
	udx = dx;
    if (dy < 0.0)
	udy = -dy;
    else
	udy = dy;

    if (udx > udy) {
	/* X major line */
	line.x_major = 1;
	line.negative = (dx < 0.0);
	line.us = FLOAT_TO_FIX_XY(v2->x);
	line.vs = FLOAT_TO_FIX_XY(v2->y);
	line.ue = FLOAT_TO_FIX_XY(v1->x);
	one_du = 1.0 / udx;
	line.dvdu = FLOAT_TO_FIX_XY(dy * one_du);
    }
    else {
	/* Y major line */
	line.x_major = 0;
	line.negative = (dy < 0.0);
	line.us = FLOAT_TO_FIX_XY(v2->y);
	line.vs = FLOAT_TO_FIX_XY(v2->x);
	line.ue = FLOAT_TO_FIX_XY(v1->y);
	one_du = 1.0 / udy;
	line.dvdu = FLOAT_TO_FIX_XY(dx * one_du);
    }

    /* Convert start Z and colors to fixed-point */
    line.zs = FLOAT_TO_FIX_Z(v2->z);
    line.rs = FLOAT_TO_FIX_RGB(v2->r);
    line.gs = FLOAT_TO_FIX_RGB(v2->g);
    line.bs = FLOAT_TO_FIX_RGB(v2->b);
    line.as = FLOAT_TO_FIX_RGB(v2->a);

    /* Compute delta values for Z and colors */
    line.dzdu = FLOAT_TO_FIX_Z((v1->z - v2->z) * one_du);
    line.drdu = FLOAT_TO_FIX_RGB((v1->r - v2->r) * one_du);
    line.dgdu = FLOAT_TO_FIX_RGB((v1->g - v2->g) * one_du);
    line.dbdu = FLOAT_TO_FIX_RGB((v1->b - v2->b) * one_du);
    line.dadu = FLOAT_TO_FIX_RGB((v1->a - v2->a) * one_du);

/* Now go draw it */

    draw_line(&line);

} /* End of setup_line */



/*
 *---------------------------------------------------------------
 *
 * draw_line
 *
 *	Draw a line.
 *
 *---------------------------------------------------------------
 */

void
draw_line(aa_setup_line *line)
{
    fix_xy x, y;			/* Start value */
    fix_xy dudu;			/* Constant 1 or -1 for step */
    fix_xy dx, dy;			/* Steps in X and Y */
    fix_z z;
    fix_rgb r, g, b, a;
    fix_xy u_off;			/* Offset to starting sample grid */
    fix_xy us, vs, ue;			/* Start and end for drawing */
    fix_xy count;			/* How many pixels to draw */
    long int color;			/* Color of generated pixel */
    long int slope_index;		/* Index into slope correction table */
    long int slope;			/* Slope correction value */
    long int ep_corr;			/* End-point correction value */
    long int scount, ecount;		/* Start/end count for endpoints */
    long int sf, ef;			/* Sand and end fractions */
    long int ep_code;			/* One of 9 endpoint codes */

    /* Get directions */
    if (line->negative)
	dudu = -ONE_XY;
    else
	dudu = ONE_XY;

    if (line->x_major) {
	dx = dudu;
	dy = line->dvdu;
    }
    else {
	dx = line->dvdu;
	dy = dudu;
    }

    /* Get initial values and count */
    if (antialiased) {
	/* Antialiased */
	if (line->negative) {
	    u_off = FRACT_XY(line->us) - ONE_XY;
	    us = line->us + ONE_XY;
	    ue = line->ue;
	    count = FLOOR_XY(us) - FLOOR_XY(ue);
	}
	else {
	    u_off = 0 - FRACT_XY(line->us);
	    us = line->us;
	    ue = line->ue + ONE_XY;
	    count = FLOOR_XY(ue) - FLOOR_XY(us);
	}
    }
    else {
	/* Jaggy */
	if (line->negative) {
	    u_off = FRACT_XY(line->us + ONEHALF_XY) - ONEHALF_XY;
	    us = FLOOR_XY(line->us + ONEHALF_XY);
	    ue = FLOOR_XY(line->ue - ONEHALF_XY);
	    count = us - ue;
	}
	else {
	    u_off = ONEHALF_XY - FRACT_XY(line->us - ONEHALF_XY);
	    us = FLOOR_XY(line->us + ONEHALF_XY);
	    ue = FLOOR_XY(line->ue + ONEHALF_XY + ONE_XY);
	    count = ue - us;
	}
    }

    vs = line->vs + fix_xy_mult(line->dvdu, u_off) + ONEHALF_XY;

    if (line->x_major) {
	x = us;
	y = vs;
    }
    else {
	x = vs;
	y = us;
    }

    z = line->zs + fix_xy_mult(line->dzdu, u_off);
    r = line->rs + fix_xy_mult(line->drdu, u_off);
    g = line->gs + fix_xy_mult(line->dgdu, u_off);
    b = line->bs + fix_xy_mult(line->dbdu, u_off);
    a = line->as + fix_xy_mult(line->dadu, u_off);

    if ((antialiased) == 0) {
	/* Jaggy line */

	/* If not capped, shorten by one */
	if (capline == 0)
	    count -= ONE_XY;

	/* Interpolate the edges */
	while ((count -= ONE_XY) >= 0) {

	    /* Now interpolate the pixels of the span */
	    color = clamp_rgb(r) |
		(clamp_rgb(g) << 8) |
		(clamp_rgb(b) << 16) |
		(clamp_rgb(a) << 24);
	    process_pixel(FIX_XY_TO_INT(x), FIX_XY_TO_INT(y),
		clamp_z(z), color);

	    x += dx;
	    y += dy;
	    z += line->dzdu;
	    r += line->drdu;
	    g += line->dgdu;
	    b += line->dbdu;
	    a += line->dadu;

	} /* End of interpolating the line parameters */
    } /* End of jaggy line code */

    else {
	/* Antialiased line */

	/* Compute slope correction once per line */
	slope_index = (line->dvdu >> (FIX_XY_SHIFT - 5)) & 0x3fu;
	if (line->dvdu < 0)
	    slope_index ^= 0x3fu;
	if ((slope_index & 0x20u) == 0)
	    slope = slope_corr_table[slope_index];
	else
	    slope = 0x100;		/* True 1.0 */

	/* Set up counters for determining endpoint regions */
	scount = 0;
	ecount = FIX_TO_INT_XY(count);

	/* Get 4-bit fractions for end-point adjustments */
	sf = (us & EP_MASK) >> EP_SHIFT;
	ef = (ue & EP_MASK) >> EP_SHIFT;

	/* Interpolate the edges */
	while (count >= 0) {

	    /*-
	     * Compute end-point code (defined as follows):
	     *  0 =  0, 0: short, no boundary crossing
	     *  1 =  0, 1: short line overlap (< 1.0)
	     *  2 =  0, 2: 1st pixel of 1st endpoint
	     *  3 =  1, 0: short line overlap (< 1.0)
	     *  4 =  1, 1: short line overlap (> 1.0)
	     *  5 =  1, 2: 2nd pixel of 1st endpoint
	     *  6 =  2, 0: last of 2nd endpoint
	     *  7 =  2, 1: first of 2nd endpoint
	     *  8 =  2, 2: regular part of line
	     */
	    ep_code = ((scount < 2) ? scount : 2) * 3 + ((ecount < 2) ? ecount : 2);
	    if (line->negative) {
		/* Drawing in the negative direction */

		/* Compute endpoint information */
		switch (ep_code) {
		  case 0: ep_corr = 0;				break;
		  case 1: ep_corr = ((sf - ef) & 0x78) | 4;	break;
		  case 2: ep_corr = sf | 4;			break;
		  case 3: ep_corr = ((sf - ef) & 0x78) | 4;	break;
		  case 4: ep_corr = ((sf - ef) + 0x80) | 4;	break;
		  case 5: ep_corr = (sf + 0x80) | 4;		break;
		  case 6: ep_corr = (0x78 - ef) | 4;		break;
		  case 7: ep_corr = ((0x78 - ef) + 0x80) | 4;	break;
		  case 8: ep_corr = 0x100;			break;
		} /* End of switch on endpoint type */
	    }
	    else {
		/* Drawing in the positive direction */

		/* Compute endpoint information */
		switch (ep_code) {
		  case 0: ep_corr = 0;				break;
		  case 1: ep_corr = ((ef - sf) & 0x78) | 4;	break;
		  case 2: ep_corr = (0x78 - sf) | 4;		break;
		  case 3: ep_corr = ((ef - sf) & 0x78) | 4;	break;
		  case 4: ep_corr = ((ef - sf) + 0x80) | 4;	break;
		  case 5: ep_corr = ((0x78 - sf) + 0x80) | 4;   break;
		  case 6: ep_corr = ef | 4;			break;
		  case 7: ep_corr = (ef + 0x80) | 4;		break;
		  case 8: ep_corr = 0x100;			break;
		} /* End of switch on endpoint type */
	    }

	    if (line->x_major)
		draw_aa_hspan(x, y, z, r, g, b, ep_corr, slope);
	    else
		draw_aa_vspan(x, y, z, r, g, b, ep_corr, slope);

	    x += dx;
	    y += dy;
	    z += line->dzdu;
	    r += line->drdu;
	    g += line->dgdu;
	    b += line->dbdu;
	    a += line->dadu;

	    scount++;
	    ecount--;
	    count -= ONE_XY;

	} /* End of interpolating the line parameters */

    } /* End of antialiased line code */

} /* End of draw_line */



/*
 *---------------------------------------------------------------
 *
 * draw_aa_hspan
 *
 *	Draw one span of an antialiased line (for horizontal lines).
 *
 *---------------------------------------------------------------
 */

void
draw_aa_hspan(fix_xy x, fix_xy y, fix_z z,
	fix_rgb r, fix_rgb g, fix_rgb b, long int ep_corr, long int slope)
{
    long int sample_dist;		/* Distance from line to sample point */
    long int filter_index;		/* Index into filter table */
    long int i;				/* Count pixels across span */
    long int index;			/* Final filter table index */
    fix_rgb a;				/* Alpha */
    long int color;			/* Final pixel color */

    sample_dist = (FRACT_XY(y) >> (FIX_XY_SHIFT - 5)) - 16;
    y = y - ONE_XY;
    filter_index = sample_dist + 32;

    for (i = 0; i < 4; i++) {
	if (filter_index < 0)
	    index = ~filter_index;	/* Invert when negative */
	else
	    index = filter_index;
	if (index > 47)
	    continue;			/* Not a valid pixel */

	a = ((((slope * ep_corr) & 0x1ff00) * filter_table[index]) &
	    0xff0000) >> 16;
	/* Should include the alpha value as well... */

	/* Draw the pixel */
	color = clamp_rgb(r) |
	    (clamp_rgb(g) << 8) |
	    (clamp_rgb(b) << 16) |
	    (a << 24);
	process_pixel(FIX_XY_TO_INT(x), FIX_XY_TO_INT(y),
	    clamp_z(z), color);

	filter_index -= 32;
	y += ONE_XY;
    }
} /* End of draw_aa_hspan */



/*
 *---------------------------------------------------------------
 *
 * draw_aa_vspan
 *
 *	Draw one span of an antialiased line (for vertical lines).
 *
 *---------------------------------------------------------------
 */

void
draw_aa_vspan(fix_xy x, fix_xy y, fix_z z,
	fix_rgb r, fix_rgb g, fix_rgb b, long int ep_corr, long int slope)
{
    long int sample_dist;		/* Distance from line to sample point */
    long int filter_index;		/* Index into filter table */
    long int i;				/* Count pixels across span */
    long int index;			/* Final filter table index */
    fix_rgb a;				/* Alpha */
    long int color;			/* Final pixel color */

    sample_dist = (FRACT_XY(x) >> (FIX_XY_SHIFT - 5)) - 16;
    x = x - ONE_XY;
    filter_index = sample_dist + 32;

    for (i = 0; i < 4; i++) {
	if (filter_index < 0)
	    index = ~filter_index;	/* Invert when negative */
	else
	    index = filter_index;
	if (index > 47)
	    continue;			/* Not a valid pixel */

	a = ((((slope * ep_corr) & 0x1ff00) * filter_table[index]) &
	    0xff0000) >> 16;
	/* Should include the alpha value as well... */

	/* Draw the pixel */
	color = clamp_rgb(r) |
	    (clamp_rgb(g) << 8) |
	    (clamp_rgb(b) << 16) |
	    (a << 24);
	process_pixel(FIX_XY_TO_INT(x), FIX_XY_TO_INT(y),
	    clamp_z(z), color);

	filter_index -= 32;
	x += ONE_XY;
    }
} /* End of draw_aa_vspan */



/*
 *---------------------------------------------------------------
 *
 * process_pixel
 *
 *	Perform blending and draw the pixel
 *
 *	Z test is shown, but not used.  Antialiased lines should
 *	be mixed with solids by drawing all solids first, checking
 *	and updating the Z-Buffer.  Antialiased lines should then
 *	be drawn checking the Z-Buffer, but not updating it.
 *
 *---------------------------------------------------------------
 */

void
process_pixel(unsigned x, unsigned y, unsigned long z, unsigned long color)
{
    int cr, cg, cb;			/* The color components */
    int ca;				/* The alpha values */
    int a1;				/* 1 - alpha */
    int or, og, ob;			/* Old RGB values */
    int old_color;			/* The old color value */
    int old_z;				/* The old Z value */
    int nr, ng, nb;			/* New RGB values */
    int new_color;			/* The new color value */
    int br, bg, bb;			/* Background color */

    if ((x >= WinWidth) || (y >= WinHeight))
	return;				/* Out of range */

    cr = color & 0xff;
    cg = (color >> 8) & 0xff;
    cb = (color >> 16) & 0xff;
    ca = (color >> 24) & 0xff;

    old_color = frame_buffer[x + y * WinWidth];
    old_z = z_buffer[x + y * WinWidth];

#if 0
    if (z > old_z)
	return;				/* Failed Z test */
#endif

    if (!antialiased) {
	draw_pixel(x, y, cr << 8, cg << 8, cb << 8);
	return;				/* No blending */
    }

    or = old_color & 0xff;
    og = (old_color >> 8) & 0xff;
    ob = (old_color >> 16) & 0xff;

    /* Blend to arbitrary background */
    if (blendmode == BLEND_ARBITRARY) {
	a1 = ca ^ 0xff;			/* 1's complement is close enough */
	nr = ((cr * ca) >> 8) + ((or * a1) >> 8);
	if (nr > 0xff)
	    nr = 0xff;			/* Clamp */
	ng = ((cg * ca) >> 8) + ((og * a1) >> 8);
	if (ng > 0xff)
	    ng = 0xff;			/* Clamp */
	nb = ((cb * ca) >> 8) + ((ob * a1) >> 8);
	if (nb > 0xff)
	    nb = 0xff;			/* Clamp */
    }
    /* Blend to constant background */
    if (blendmode == BLEND_CONSTANT) {
	br = background & 0xff;		/* Sorry this isn't optimized */
	bg = (background >> 8) & 0xff;
	bb = (background >> 16) & 0xff;

	nr = (((cr - br) * ca) >> 8) + or;
	if (nr > 0xff)
	    nr = 0xff;			/* Clamp */
	if (nr < 0)
	    nr = 0;
	ng = (((cg - bg) * ca) >> 8) + og;
	if (ng > 0xff)
	    ng = 0xff;			/* Clamp */
	if (ng < 0)
	    ng = 0;
	nb = (((cb - bb) * ca) >> 8) + ob;
	if (nb > 0xff)
	    nb = 0xff;			/* Clamp */
	if (nb < 0)
	    nb = 0;
    }

    /* Add to background */
    if (blendmode == ADD_TO_BACKGROUND) {
	nr = ((cr * ca) >> 8) + or;
	if (nr > 0xff)
	    nr = 0xff;			/* Clamp */
	ng = ((cg * ca) >> 8) + og;
	if (ng > 0xff)
	    ng = 0xff;			/* Clamp */
	nb = ((cb * ca) >> 8) + ob;
	if (nb > 0xff)
	    nb = 0xff;			/* Clamp */
    }

    new_color = nr | (ng << 8) | (nb << 16);
    frame_buffer[x + y * WinWidth] = new_color;

    draw_pixel(x, y, nr << 8, ng << 8, nb << 8);

} /* End of process_pixel */



/*
 *---------------------------------------------------------------
 *
 * fix_xy_mult
 *
 *	Multiply a fixed-point number by a s11.20 fixed-point
 *	number.  The actual multiply uses less bits for the
 *	multiplier, since it always represents a fraction
 *	less than 1.0 and less total bits are sufficient.
 *	Some of the steps here are not needed.  This was originally
 *	written to simulate exact hardware behavior.
 *
 *	This could easily be optimized when using a flexible compiler.
 *
 *---------------------------------------------------------------
 */

long int
fix_xy_mult(long int a, fix_xy b)
{
    int negative;			/* 1 = result is negative */
    int a1;				/* Multiplier */
    int bh, bl;				/* Multiplicant (high and low) */
    int ch, cl, c;			/* Product */

    /* Determine the sign, then force multiply to be unsigned */
    negative = 0;
    if (a < 0) {
	negative ^= 1;
	a = -a;
    }
    if (b < 0) {
	negative ^= 1;
	b = -b;
    }

    /* Grab the bits we want to use */
    a1 = a >> 10;			/* Just use 10-bit fraction */

    /* Split the 32-bit number into two 16-bit halves */
    bh = (b >> 16) & 0xffff;
    bl = b & 0xffff;

    /* Perform the multiply */
    ch = bh * a1;			/* 30 bit product (with no carry) */
    cl = bl * a1;
    /* Put the halves back together again */
    c = (ch << 6) + (cl >> 10);
    if (negative)
	c = -c;

    return c;
} /* End of fix_xy_mult */



/*
 *---------------------------------------------------------------
 *
 * clamp_rgb
 *
 *	Clamp a fixed-point color value and return it as an 8-bit value.
 *
 *---------------------------------------------------------------
 */

long int
clamp_rgb(long int x)
{
    if (x < 0)
	x = 0;
    else if (x >= ONE_RGB)
	x = ONE_RGB - 1;

    return (x >> (30 - 8));
} /* End of clamp_rgb */



/*
 *---------------------------------------------------------------
 *
 * clamp_z
 *
 *	Clamp a fixed-point Z value and return it as a 28-bit value.
 *
 *---------------------------------------------------------------
 */

long int
clamp_z(long int x)
{
    if (x < 0)
	x = 0;
    else if (x >= ONE_Z)
	x = ONE_Z - 1;

    return (x >> (30 - 28));
} /* End of clamp_z */



/*
 *---------------------------------------------------------------
 *
 * init_tables
 *
 *	Initialize the tables normally found in ROM in the hardware.
 *
 *---------------------------------------------------------------
 */

void
init_tables(void)
{
    int i;				/* Iterative counter */
    double m;				/* Slope */
    double d;				/* Distance from center of curve */
    double v;				/* Value to put in table */
    double sr;				/* The square root value */
    long int *fb_ptr;			/* Used in clearing the frame buffer */
    long int *fb_end;

    /*-
     * Build slope correction table.  The index into this table
     * is the truncated 5-bit fraction of the slope used to draw
     * the line.  Round the computed values here to get the closest
     * fit for all slopes matching an entry.
     */

    for (i = 0; i < SC_TABLE_SIZE; i++) {
	/* Round and make a fraction */
	m = ((double) i + 0.5) / (float) SC_TABLE_SIZE;
	v = sqrt(m * m + 1) * 0.707106781; /* (m + 1)^2 / sqrt(2) */
	slope_corr_table[i] = (long int) (v * 256.0);
    }

    /*-
     * Build the Gaussian filter table, round to the middle of the
     * sample region.
     */

    for (i = 0; i < F_TABLE_SIZE; i++) {
	d = ((double) i + 0.5) / (float) (F_TABLE_SIZE / 2.0);
	d = d / FILTER_WIDTH;
	v = 1.0 / exp(d * d);		/* Gaussian function */
	filter_table[i] = (long int) (v * 256.0);
    }
    /*-
     * Build the square root table for big dots.
     */

    for (i = 0; i < SR_TABLE_SIZE; i++) {
	v = (double) ((i << 1) + 1) / (double) (1 << (SRT_FRACT + 1));
	sr = sqrt(v);
	sqrt_table[i] = (long int) (sr * (double) (1 << SR_FRACT));
    }

/* Allocate and clear the frame buffer too */

    frame_buffer = (long int *)malloc(WinWidth * WinHeight * sizeof(long int));
    if (frame_buffer == 0)
	fprintf(stderr, "Failed to allocate frame_buffer\n"), exit(0);
    fb_ptr = frame_buffer;
    fb_end = fb_ptr + (WinWidth * WinHeight);
    while (fb_ptr < fb_end)
	*fb_ptr++ = background;

    z_buffer = (long int *)malloc(WinWidth * WinHeight * sizeof(long int));
    if (z_buffer == 0)
	fprintf(stderr, "Failed to allocate z_buffer\n"), exit(0);
    fb_ptr = z_buffer;
    fb_end = fb_ptr + (WinWidth * WinHeight);
    while (fb_ptr < fb_end)
	*fb_ptr++ = 0xffffffff;

} /* End of init_tables */

/* End of aadraw.c */
