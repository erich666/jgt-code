DEBUG = -O2
#DEBUG = -g
CXXFLAGS += $(DEBUG)
LIBS += -lm

SHAREDLIB = libsubdeval.so

ALL = $(SHAREDLIB)

OBJS = Subd.o geom.o
HEADERS = IntPairMap.h Subd.h SubdEigenData.h  geom.h

%.o: %.C $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(INC)

all : $(ALL)

$(SHAREDLIB) : $(OBJS)
	$(CXX) -shared -fPIC $^ -o $@ $(CXXFLAGS) $(LIBS)

clean:
	rm *.o $(SHAREDLIB)
