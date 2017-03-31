CXX=g++ -O3 -s -Wall
exe: exe.o cylinder.o axis.o vector.o cone.o spherecone.o spherecylinder.o
	$(CXX) $(CXXFLAGS) -o exe exe.o cylinder.o cone.o axis.o vector.o spherecone.o spherecylinder.o

exe.o: exe.c shapes.h
	$(CXX) $(CXXFLAGS) -c exe.c

cylinder.o: cylinder.c shapes.h  
	$(CXX) $(CXXFLAGS) -c cylinder.c
				
cone.o: cone.c shapes.h  
	$(CXX) $(CXXFLAGS) -c cone.c
				
axis.o: axis.c shapes.h  
	$(CXX) $(CXXFLAGS) -c axis.c

spherecone.o: spherecone.c shapes.h  
	$(CXX) $(CXXFLAGS) -c spherecone.c

spherecylinder.o: spherecylinder.c shapes.h  
	$(CXX) $(CXXFLAGS) -c spherecylinder.c

vector.o: vector.c vector.h  
	$(CXX) $(CXXFLAGS) -c vector.c
# Clean
clean:
	rm -r -f *.o exe Html
doc:
	doxygen Cone.oxy
	chmod u+rw-x,og-xw+r Html/*
