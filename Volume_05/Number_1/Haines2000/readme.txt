Code for shaft-culling paper,
by Eric Haines, erich@acm.org

This code can be reused freely, commercially or not. No GPL, no nothing: I
consider it placed in the public domain.

The files you need to include shaft culling in your code:

shaft.c - the algorithm code.
shaft.h - the external include file to put in your calling code.
shafttab.h - a look-up table for forming planes, internal include
	file used only if USE_TABLE is defined.
main.c - a test program for the shaft code.
makefile - generic makefile (ignores shafttab.h).
shaft.ds* - MS VC++ makefiles.
readme.txt - this file.


See shaft.h for full documentation on how to use the shaft code. See main.c
for various testing options.