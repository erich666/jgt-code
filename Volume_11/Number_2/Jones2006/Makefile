CFLAGS=`glib-config --cflags` -g -O -Wall 

fast_delaunay:	fast_delaunay.o  weighted_tree.o voronoi.o clip.o triangle_area.o
	$(CC) -o $@ $^ `gts-config --libs` -lm
