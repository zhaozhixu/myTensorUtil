.SUFFIXES:
CC = g++
CUCC = nvcc

CFLAGS = -std=c++11 -Wall
CUFLAGS = -m64 -ccbin $(CC)
LDFLAGS = $(CFLAGS)

ifdef DEBUG
CFLAGS += -g -O0 -DDEBUG
CUFLAGS += -lineinfo
LDFLAGS += -g -O0
else
CFLAGS += -O3 -DNDEBUG
CUFLAGS +=
LDFLAGS += -O3
endif

ifdef VERBOSE
AT=
else
AT=@
endif

AR = ar cr
ECHO = @echo
SHELL=/bin/sh

TRIPLE?=x86_64-linux
CUDA_INSTALL_DIR = /usr/local/cuda-8.0
CUDA_LIBDIR = lib
INCPATHS    =-I"$(CUDA_INSTALL_DIR)/include" -I"/usr/local/include"
LIBPATHS    =-L"$(CUDA_INSTALL_DIR)/targets/$(TRIPLE)/$(CUDA_LIBDIR)" -L"/usr/local/lib" -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)"
LIBS = -lcudart -lcudart_static
CFLAGS += $(INCPATHS)
CUFLAGS += $(INCPATHS)
LDFLAGS += $(LIBPATHS)

define concat
  $1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(CFLAGS) $1
endef

define make-depend-cu
  $(AT)$(CUCC) -M $(CUFLAGS) $1 > $3.$$$$; \
  sed 's,.*\.o[ :]*,$2 : ,g' < $3.$$$$ > $3; \
  rm -f $3.$$$$
endef

SRCS_C = test.c tensorUtil.cu errorHandle.cu sdt_alloc.c
TARGET = testtu
OUTDIR = .
OBJDIR = $(call concat,$(OUTDIR),/obj)
OBJS   = $(patsubst %.c, $(OBJDIR)/%.o, $(wildcard *.c))
OBJS  += $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard *.cpp))
CUOBJS = $(patsubst %.cu, $(OBJDIR)/%.o, $(wildcard *.cu))

.PHONY: all
all: $(TARGET)

$(OUTDIR)/$(TARGET): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $<
	$(AT)$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: %.c
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-cu,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR)

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(OBJDIR)/*.d
endif
