include /home/mmousa/CGAL-3.2/make/makefile_i686_Linux-2.6_g++-4.0.1
#CGAL_LDFLAGS CGAL_CXXFLAGS
LIB = $(CGAL_LDFLAGS) -lgsl -lgslcblas 
INC = $(CGAL_CXXFLAGS) 

harmonics : main.o misc.o cmodel.o integration.o
	$(CGAL_CXX) -o harmonics main.o misc.o cmodel.o integration.o $(LIB) 

main.o : main.cpp config.h
	$(CGAL_CXX) -c main.cpp $(INC)

misc.o : misc.h misc.cpp config.h
	$(CGAL_CXX) -c misc.cpp $(INC)

cmodel.o : cmodel.h cmodel.cpp config.h
	$(CGAL_CXX) -c cmodel.cpp $(INC)

fst.o : fst.h fst.cpp 
	$(CGAL_CXX) -c fst.cpp $(INC) 

integration.o : integration.h integration.cpp config.h
	$(CGAL_CXX) -c integration.cpp $(INC)
#################################################################
clean : 
	rm -f *.o
