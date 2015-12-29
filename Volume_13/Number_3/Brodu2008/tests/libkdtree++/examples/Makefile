CXXFLAGS = -Wall -W -pedantic -ansi
LDFLAGS = -lstdc++
test_kdtree: test_kdtree.o

CC = $(CXX)

ifeq "$(wildcard ../include/kdtree.hpp)" ""
CPPFLAGS=`pkg-config libkdtree++ --cflags`
else
INCLUDE = .$(shell ln -sf ../include kdtree++)
CPPFLAGS = -I$(INCLUDE)
CXXFLAGS += -O0 -g
test_kdtree.o: $(wildcard $(INCLUDE)/kdtree++/*.hpp)
endif

.PHONY: run
run:: test_kdtree
	@./$<

.PHONY: debug
debug:: test_kdtree
	@gdb -q ./$<

.PHONY: clean
clean::
	rm -f $(wildcard *.o) test_kdtree kdtree++
