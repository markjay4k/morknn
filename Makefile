CXX = nvcc
CXXFLAGS = -arch=sm_61 --shared -Xcompiler -fPIC -I/usr/include/python3.11

matmul.so: matmul.cu
	$(CXX) $(CXXFLAGS) -o matmul.so matmul.cu 

clean:
	rm -f matmul.so

