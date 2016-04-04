/*
 * version.c
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
 * $Id: version.c,v 4.0 91/07/17 14:52:08 kolb Exp Locker: kolb $
 *
 * $Log:	version.c,v $
 * Revision 4.0  91/07/17  14:52:08  kolb
 * Initial version.
 * 
 */
#include "rayshade.h"
#include "patchlevel.h"
#include "stats.h"

void
VersionPrint()
{
	extern char rcsid[];

	fprintf(Stats.fstats,
		"rayshade: %s\nPatch level %d\n", rcsid, PATCHLEVEL);
}
