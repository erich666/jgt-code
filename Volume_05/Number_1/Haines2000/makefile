.SUFFIXES: .c .o .h

.c.o:
	$(CC) -c $<

shaft : main.o shaft.o
	$(CC) -o shaft main.o shaft.o

