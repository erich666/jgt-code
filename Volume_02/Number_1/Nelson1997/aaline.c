/*
 * aaline.c
 *
 *	Test program to draw antialiased lines.
 *
 *	This module contains the main program.
 *	Written in a machine independent manner
 *	requiring only one machine specific module
 *	to write pixels to the screen.
 */

#include <math.h>
#include "aaline.h"



/* Local function prototypes */
void test1(void);



/* Global flags used by this code */
int antialiased = 1;		/* Antialiased lines, not jaggy */
int capline = 1;		/* Draw last pixel of jaggy lines */
int blendmode = BLEND_CONSTANT;	/* Which blend mode to use */
int background = 0x000000;	/* Background color (bbggrr), 0 = black */



/*
 *---------------------------------------------------------------
 *
 * Main program
 *
 *---------------------------------------------------------------
 */

void
main(void)
{

    init_screen(background);		/* Prepare to draw to screen */

    init_tables();			/* Initialize lookup tables */

    test1();				/* Draw a test image */

    close_screen();
} /* End of main */



/*
 *---------------------------------------------------------------
 *
 * test1
 *
 *	Draw some test objects that show good line behavior.
 *
 *---------------------------------------------------------------
 */

void
test1(void)
{
    long int x, y;
    aa_vertex v1, v2;
    float cx, cy;			/* Center X and Y */
    float sx, sy;			/* Scale X and Y */
    float nx, ny;			/* New X and Y, from trig functions */
    float x1, y1, x2, y2;		/* Computed values */
    float angle;

    /* Initialize the colors */
    v1.z = 0.5;
    v1.r = 0.75;			/* Almost white */
    v1.g = 0.75;
    v1.b = 0.75;
    v1.a = 1.0;

    v2.z = 0.5;
    v2.r = 0.75;
    v2.g = 0.75;
    v2.b = 0.75;
    v2.a = 1.0;

   /* Radial lines, in to out */
    cx = 150.2;
    cy = 150.2;
    sx = 125.0;
    sy = 125.0;
    for (angle = 0.0; angle < 360.0; angle += 2.0) {
	nx = sin(angle * 3.1415926535 / 180.0);
	ny = cos(angle * 3.1415926535 / 180.0);
	x1 = cx;
	y1 = cy;
	x2 = nx * sx + cx;
	y2 = ny * sy + cy;
	v1.x = x1;
	v1.y = y1;
	v2.x = x2;
	v2.y = y2;
	setup_line(&v1, &v2);
    }

    /* Radial lines, out to in, change color along lines */
    v1.r = 0.0;				/* Blue center */
    v1.g = 0.0;
    v1.b = 0.5;
    v2.r = 1.0;				/* Yellow edge */
    v2.g = 1.0;
    v2.b = 0.0;
    cx = 400.2;
    cy = 150.2;
    sx = 125.0;
    sy = 125.0;
    for (angle = 0.0; angle < 360.0; angle += 2.0) {
	nx = sin(angle * 3.1415926535 / 180.0);
	ny = cos(angle * 3.1415926535 / 180.0);
	x1 = cx;
	y1 = cy;
	x2 = nx * sx + cx;
	y2 = ny * sy + cy;
	v1.x = x1;
	v1.y = y1;
	v2.x = x2;
	v2.y = y2;
	setup_line(&v2, &v1);
    }

    /* Radial lines, with saturation */
    v1.r = 1.0;				/* Orange */
    v1.g = 0.4;
    v1.b = 0.1;
    v2.r = 1.0;
    v2.g = 0.4;
    v2.b = 0.1;
    cx = 570.0;
    cy = 70.0;
    sx = 65.0;
    sy = 65.0;
    for (angle = 0.0; angle < 360.0; angle += 2.0) {
	nx = sin(angle * 3.1415926535 / 180.0);
	ny = cos(angle * 3.1415926535 / 180.0);
	x1 = cx;
	y1 = cy;
	x2 = nx * sx + cx;
	y2 = ny * sy + cy;
	v1.x = x1;
	v1.y = y1;
	v2.x = x2;
	v2.y = y2;
	setup_line(&v2, &v1);
    }

    /* Concentric circles */
    v1.r = 0.5;				/* Medium grey */
    v1.g = 0.5;
    v1.b = 0.5;
    v2.r = 0.5;
    v2.g = 0.5;
    v2.b = 0.5;
    cx = 105.0;
    cy = 375.0;
    for (sx = 90.0; sx > 40.0; sx *= 0.97) {
	sy = sx;
	x1 = sin(0.0) * sx + cx;
	y1 = cos(0.0) * sy + cy;
	for (angle = 0.0; angle < 361.0; angle += 2.0) {
	    nx = sin(angle * 3.1415926535 / 180.0);
	    ny = cos(angle * 3.1415926535 / 180.0);
	    x2 = nx * sx + cx;
	    y2 = ny * sy + cy;
	    v1.x = x1;
	    v1.y = y1;
	    v2.x = x2;
	    v2.y = y2;
	    x1 = x2;
	    y1 = y2;
	    setup_line(&v2, &v1);
	}
    }

    /* Small circles */
    v1.r = 0.25;			/* Light blue */
    v1.g = 0.5;
    v1.b = 1.0;
    v2.r = 0.25;
    v2.g = 0.5;
    v2.b = 1.0;
    cx = 250.0;
    cy = 310.0;
    for (sx = 12.0; sx > 0.5; sx *= 0.93) {
	sy = sx;
	x1 = sin(0.0) * sx + cx;
	y1 = cos(0.0) * sy + cy;
	for (angle = 0.0; angle < 361.0; angle += 2.0) {
	    nx = sin(angle * 3.1415926535 / 180.0);
	    ny = cos(angle * 3.1415926535 / 180.0);
	    x2 = nx * sx + cx;
	    y2 = ny * sy + cy;
	    v1.x = x1;
	    v1.y = y1;
	    v2.x = x2;
	    v2.y = y2;
	    x1 = x2;
	    y1 = y2;
	    setup_line(&v2, &v1);
	}
	cx += sx * 2.0 + 4.0;
	if (cx > 500) {
	    cx = 250.0;
	    cy += sy * 2.0 + 12.0;
	}
    }
} /* End of test1 */

/* End of aaline.c */
