CXX := g++
CXXFLAGS := -g -O3 -march=native -fopenmp -std=c++17
TARGET := main
SOURCES := $(wildcard src/*.cpp)
MATRIX_SOURCES := src/matrix_multiply.cpp
TEST_TARGET := test/test_matrix_multiply
TEST_SOURCE := test/test_matrix_multiply.cpp

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

$(TEST_TARGET): $(TEST_SOURCE) $(MATRIX_SOURCES) src/matrix_multiply.h src/aligned_allocator.h
	$(CXX) $(CXXFLAGS) -Isrc $(TEST_SOURCE) $(MATRIX_SOURCES) -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET)
