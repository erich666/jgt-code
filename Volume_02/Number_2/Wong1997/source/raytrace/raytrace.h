/*
 * raytrace.h
 *
 * Copyright (C) 1989, 1991, Craig E. Kolb
 * All rights reserved.
 *
 * This software may be freely copied, modified, and redistributed
 * provided that this copyright notice is preserved on all copies.
 *
 * You may not distribute this software, in whole or in part, as part of
 * any commercial product without the express consent of the authors.
 *
 * There is no warranty or other guarantee of fitness of this software
 * for any purpose.  It is provided solely "as is".
 *
 * $Id: raytrace.h,v 4.0 91/07/17 14:51:25 kolb Exp Locker: kolb $
 *
 * $Log:	raytrace.h,v $
 * Revision 4.0  91/07/17  14:51:25  kolb
 * Initial version.
 * 
 */
#ifndef RAYTRACE_H
#define RAYTRACE_H

#include "picture.h"

typedef struct {
	Float x, y, size;	/* location and size */
	Pixel ul, ur, ll, lr;	/* Color values of four corners */
} pixel_square;

extern void		raytrace();

#endif /* RAYTRACE_H */
