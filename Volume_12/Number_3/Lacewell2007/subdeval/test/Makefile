DEBUG = -O2
CXXFLAGS += $(DEBUG)

INC += -I../
LIBS += -L../ -lsubdeval -Wl,-rpath=../

ALL = testsubd

OBJS = testsubd.o
HEADERS =

%.o: %.C
	$(CXX) -c -o $@ $< $(CXXFLAGS) $(INC)

all : $(ALL)

testsubd: $(OBJS) $(HEADERS)
	$(CXX) $(OBJS) -o $@ $(CXXFLAGS) $(LIBS)

clean:
	rm *.o testsubd
