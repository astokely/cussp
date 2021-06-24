# standard compile options for the c++ executable
FLAGS = -fPIC

# the python interface through swig
PYTHONI = -I/home/astokely/miniconda3/include/python3.8/
NUMPYI = -I/home/astokely/miniconda3/lib/python3.8/site-packages/numpy/core/include/
PYTHONL = -Xlinker -export-dynamic

all: program

program: cussp.o
	swig -c++ -python -o cussp/cussp_wrap.cxx cussp/cussp.i 
	g++  $(FLAGS) $(PYTHONI) $(NUMPYI) -c cussp/cussp_wrap.cxx -o cussp/cussp_wrap.o 
	g++  $(PYTHONL) $(LIBFLAGS) -shared cussp/cussp.o cussp/cussp_cuda.o cussp/cussp_wrap.o -o cussp/_cussp.so -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include

cussp_cuda.o:
	nvcc -c -Xcompiler -fPIC cuda/kernels/cussp.cu -o cussp/cussp_cuda.o

cussp.o: cussp_cuda.o
	g++ -fPIC -c cuda/cussp.cpp -o cussp/cussp.o 

clean:	
	$(RM) -rf cussp/cussp.py cussp/*cxx* cussp/*.so cussp/*.o cussp/__pycache__
