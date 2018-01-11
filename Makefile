all: first.o liblbfgs-1.10/lib/.libs/lbfgs.o
	g++ first.o liblbfgs-1.10/lib/.libs/lbfgs.o -o first.out

first.o: first.cpp Tensor.cpp
	g++ -I liblbfgs-1.10/include -c first.cpp   -o first.o 

test:
	./first.out 1 1 0.001
