CC = g++
CUCC = nvcc -m64
TARGET = testtu

ifeq ($(DEBUG),1)
CC += -g -O0 -DDEBUG
CUCC += -lineinfo -ccbin $(CC)
else
CC += -O3 -DNDEBUG
CUCC += -ccbin $(CC)
endif

TRIPLE?=x86_64-linux
CUDA_INSTALL_DIR = /usr/local/cuda-8.0
CUDA_LIBDIR = lib

INCPATHS    =-I"$(CUDA_INSTALL_DIR)/include" -I"/usr/local/include" -I".."
LIBPATHS    =-L"$(CUDA_INSTALL_DIR)/targets/$(TRIPLE)/$(CUDA_LIBDIR)" -L"/usr/local/lib" -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)"
COMMON_LIBS = -lcudart -lcudart_static `pkg-config --cflags --libs opencv` -std=c++11

# VPATH = ..

# $(TARGET): test.o tensorUtil.o tensorCuda.o trtUtil.o errorHandle.o
# 	$(CC) -Wall -g test.o trtUtil.o tensorUtil.o tensorCuda.o errorHandle.o -o testtrt $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
$(TARGET): test.o tensorUtil.o errorHandle.o sdt_alloc.o
	$(CC) -Wall test.o tensorUtil.o errorHandle.o sdt_alloc.o -o $(TARGET) $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
test.o: test.c tensorUtil.h errorHandle.h sdt_alloc.h
	$(CC) -c test.c $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
# trtUtil.o: trtUtil.cpp trtUtil.h errorHandle.h
# 	$(CUCC) -c trtUtil.cpp $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
tensorUtil.o: tensorUtil.cu tensorUtil.h errorHandle.h sdt_alloc.h
	$(CUCC) -c tensorUtil.cu $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
# tensorCuda.o: tensorCuda.cu tensorCuda.h errorHandle.h
# 	$(CUCC) -g -c tensorCuda.cu $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
errorHandle.o: errorHandle.cu errorHandle.h
	$(CUCC) -c errorHandle.cu $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
sdt_alloc.o: sdt_alloc.c sdt_alloc.h
	$(CC) -c sdt_alloc.c $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
clean:
	rm -f *.o
