/*
 * aaline.h
 *
 *	Structure definitions, function prototypes, and
 *	so forth for the antialiased line drawing sample code.
 *
 *	This code assumes integers are 32 bits.
 */

#include <stdio.h>
#include <stdlib.h>

/* Definitions used in this code */
#define WinWidth	640
#define WinHeight	480

/* Blend modes */
#define BLEND_ARBITRARY	1	/* Blend to arbitrary background color */
#define BLEND_CONSTANT	2	/* Blend to constant background color */
#define ADD_TO_BACKGROUND 3	/* Add to background color */



/* Frame Buffer Data Structure */

typedef struct fb fb;

struct fb {
    int fb_width;		/* Number of  horizontal pixels */
    int fb_height;		/* Number of vertical pixels */
    unsigned long *fbptr;	/* Pointer to frame buffer memory */
};


/* Data used when setting up a line.  Should be in a context structure */

/* Convert from floating-point to internal fixed-point formats */
#define ONE_XY			(long int) 0x00100000
#define FIX_XY_SHIFT		(long int) 20
#define ONEHALF_XY		(long int) 0x00080000
#define ONE_Z			(long int) 0x40000000
#define ONE_RGB			(long int) 0x40000000
#define ONE_16			(long int) 0x4000

#define FLOAT_TO_FIX_XY(x)	((long int) ((x) * (float) ONE_XY))

#define FLOAT_TO_FIX_Z(x)	((long int) ((x) * (float) ONE_Z))
#define FLOAT_TO_FIX_RGB(x)	((long int) ((x) * (float) ONE_RGB))
#define FLOAT_TO_FIX_16(x)	((long int) ((x) * (float) ONE_16))
#define FIX_TO_INT_XY(x)	((x) >> FIX_XY_SHIFT)
#define FIX_16_TO_FLOAT(x)	((float) (x) / (float) ONE_16)
#define FIX_TO_FLOAT_XY(x)	((float) (x) / (float) ONE_XY)
#define FIX_TO_FLOAT_Z(x)	((float) (x) / (float) ONE_Z)
#define FIX_TO_FLOAT_RGB(x)	((float) (x) / (float) ONE_RGB)

/* Get fractional part, next lowest integer part */
#define FRACT_XY(x)		((x) & (long int) 0x000fffff)
#define FLOOR_XY(x)		((x) & (long int) 0xfff00000)
#define FIX_XY_TO_INT(x)	((long int) (x) >> (long int) FIX_XY_SHIFT)

/* Sizes for tables in Draw */
#define FILTER_WIDTH	0.75	/* Line filter width adjustment */
#define F_TABLE_SIZE	64	/* Filter table size */
#define SC_TABLE_SIZE	32	/* Slope correction table size */
#define SRT_INT		5	/* Sqrt table index integer bits */
#define SRT_FRACT	4	/* ...fraction bits */
#define SR_INT		3	/* Square root result integer bits */
#define SR_FRACT	5	/* ...fraction bits */
#define SR_TABLE_SIZE	(1 << (SRT_INT + SRT_FRACT))

#define EP_MASK		(long int) 0x000f0000u	/* AA line end-point filter mask */
#define EP_SHIFT	13u	/* Number of bits to shift end-point */


typedef long int fix_xy;	/* S11.20 */
typedef long int fix_z;		/* S1.30 */
typedef long int fix_rgb;	/* S1.30 */

/* One vertex at any of the various stages of the pipeline */
typedef struct aa_vertex aa_vertex;
struct aa_vertex {
    unsigned header;
    float x, y, z, w;
    float r, g, b, a;
};

/* All values needed to draw one line */
typedef struct aa_setup_line aa_setup_line;
struct aa_setup_line {
    int x_major;
    int negative;

    fix_xy vs;			/* Starting point */
    fix_xy us;
    fix_xy ue;			/* End (along major axis) */
    fix_xy dvdu;		/* Delta for minor axis step */

    fix_z zs;			/* Starting Z and color */
    fix_rgb rs;
    fix_rgb gs;
    fix_rgb bs;
    fix_rgb as;

    fix_z dzdu;			/* Delta for Z and color */
    fix_rgb drdu;
    fix_rgb dgdu;
    fix_rgb dbdu;
    fix_rgb dadu;
};



/* Global variables */
extern int antialiased;		/* Antialiased lines, not jaggy */
extern int capline;		/* Draw last pixel of jaggy lines */
extern int blendmode;		/* Which blend mode to use */
extern int background;		/* Black */

/* Function prototypes */
void init_screen(int background);
void draw_pixel(long int x, long int y, short r, short g, short b);
void close_screen(void);
void init_tables(void);

void setup_line(aa_vertex *v1, aa_vertex *v2);
void draw_line(aa_setup_line *line);
void draw_aa_hspan(fix_xy x, fix_xy y, fix_z z,
    fix_rgb r, fix_rgb g, fix_rgb b, long int ep_corr, long int slope);
void draw_aa_vspan(fix_xy x, fix_xy y, fix_z z,
    fix_rgb r, fix_rgb g, fix_rgb b, long int ep_corr, long int slope);
long int fix_xy_mult(long int a, fix_xy b);
long int clamp_rgb(long int x);
long int clamp_z(long int x);
long int float_to_fix(float x, long int one);

void process_pixel(unsigned x, unsigned y, unsigned long z, unsigned long color);


/* End of aaline.h */
