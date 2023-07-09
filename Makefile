CXX = nvcc
CXXFLAGS = -arch=sm_80 --shared -Xcompiler -fPIC -I/usr/include/python3.11
CUDA_DIR = ./morknn/cuda

matmul.so: $(CUDA_DIR)/matmul.cu
	$(CXX) $(CXXFLAGS) -use_fast_math -O2 -o $(CUDA_DIR)/matmul.so $(CUDA_DIR)/matmul.cu 

clean:
	rm -f $(CUDA_DIR)/matmul.so

