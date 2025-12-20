CXX = mpicxx
CXXFLAGS = -O3 -fopenmp -Wall
TARGET = poisson_solver

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)