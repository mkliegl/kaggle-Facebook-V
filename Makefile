CXX=clang++
#CXX=g++-6

CXXFLAGS=-Wall -O3 -std=c++14

CXXPROGS := $(patsubst %.cc,%,$(wildcard *.cc))

.PHONY: all
all: $(CXXPROGS)

.PHONY: check
check:
	cppcheck --enable=all --suppress=missingIncludeSystem \
             --inconclusive --std=c++11 \
             singlepass.cc

singlepass: singlepass.cc
	$(CXX) $(CXXFLAGS) -o $@ $<
