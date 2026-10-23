CXX ?= g++
CXXFLAGS ?= -O3 -march=native -ffast-math -fno-finite-math-only -fopenmp -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpng -lz

slicer: slicer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f slicer
