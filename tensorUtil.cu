#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
/* #include "tensorCuda.h" */
#include "tensorUtil.h"
#include "errorHandle.h"
#include "sdt_alloc.h"

#define MAXDIM 8
#define MAX_THREADS_PER_BLOCK 1024
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

/* static float EPSILON = 1e-16; */

static void assertTensor(const Tensor *tensor)
{
     assert(tensor && tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0);
     assert(tensor->len == computeLength(tensor->ndim, tensor->dims));
}

int isTensorValid(const Tensor *tensor)
{
     return (tensor && tensor->data &&
             tensor->ndim < MAXDIM && tensor->ndim > 0 &&
             tensor->len == computeLength(tensor->ndim, tensor->dims));
}

int isShapeEqual(const Tensor *t1, const Tensor *t2)
{
     assertTensor(t1);
     assertTensor(t2);
     if (t1->ndim == t2->ndim) {
          int ndim = t1->ndim;
          while (--ndim >= 0)
               if (t1->dims[ndim] != t2->dims[ndim])
                    return 0;
          return 1;
     }
     return 0;
}

/* can only identify host memory alloced by cudaMallocHost, etc */
int isHostMem(const void *ptr)
{
     cudaPointerAttributes attributes;
     checkError(cudaPointerGetAttributes(&attributes, ptr));
     return attributes.memoryType == cudaMemoryTypeHost;
}

int isDeviceMem(const void *ptr)
{
     cudaPointerAttributes attributes;
     checkError(cudaPointerGetAttributes(&attributes, ptr));
     return attributes.memoryType == cudaMemoryTypeDevice;
}

void *cloneMem(const void *src, size_t size, CloneKind kind)
{
     assert(src);
     void *p;
     switch (kind) {
     case H2H:
          p = sdt_alloc(size);
          memmove(p, src, size);
          return p;
     case H2D:
          checkError(cudaMalloc(&p, size));
          checkError(cudaMemcpy(p, src, size, cudaMemcpyHostToDevice));
          return p;
     case D2D:
          assert(isDeviceMem(src));
          checkError(cudaMalloc(&p, size));
          checkError(cudaMemcpy(p, src, size, cudaMemcpyDeviceToDevice));
          return p;
     case D2H:
          assert(isDeviceMem(src));
          p = sdt_alloc(size);
          checkError(cudaMemcpy(p, src, size, cudaMemcpyDeviceToHost));
          return p;
     default:
          fprintf(stderr, "unknown CloneKind %d\n", kind);
          return NULL;
     }

}

Tensor *cloneTensor(const Tensor *src, CloneKind kind)
{
     assert(isTensorValid(src));
     float *data = (float *)cloneMem(src->data, src->len * sizeof(float), kind);
     Tensor *dst = createTensor(data, src->ndim, src->dims);
     return dst;
}

void *repeatMem(void *data, size_t size, int times, CloneKind kind)
{
     assert(data && times > 0);
     void *p, *dst;
     int i;
     switch (kind) {
     case H2H:
          dst = p = sdt_alloc(size * times);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               memmove(p, data, size);
          return dst;
     case H2D:
          checkError(cudaMalloc(&p, size * times));
          dst = p;
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               checkError(cudaMemcpy(p, data, size, cudaMemcpyHostToDevice));
          return dst;
     case D2D:
          assert(isDeviceMem(data));
          checkError(cudaMalloc(&p, size * times));
          dst = p;
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               checkError(cudaMemcpy(p, data, size, cudaMemcpyDeviceToDevice));
          return dst;
     case D2H:
          assert(isDeviceMem(data));
          dst = p = sdt_alloc(size * times);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               checkError(cudaMemcpy(p, data, size, cudaMemcpyDeviceToHost));
          return dst;
     default:
          fprintf(stderr, "unknown CloneKind %d\n", kind);
          return NULL;
     }
}


int computeLength(int ndim, const int *dims)
{
     if (dims) {
          int i, len = 1;
          for (i = 0; i < ndim; i++)
               len *= dims[i];
          return len;
     }
     fprintf(stderr, "Warning: null dims in computeLength\n");
     return 0;
}

Tensor *createTensor(float *data, int ndim, const int *dims)
{
     Tensor *t = (Tensor *)sdt_alloc(sizeof(Tensor));
     t->data = data;
     t->ndim = ndim;
     t->dims = (int *)sdt_alloc(sizeof(int) * ndim);
     memmove(t->dims, dims, sizeof(int) * ndim);
     t->len = computeLength(ndim, dims);
     return t;
}

Tensor *mallocTensor(int ndim, const int* dims, const MallocKind mkind)
{
     Tensor *t = createTensor(NULL, ndim, dims);
     float *f;

     switch (mkind) {
     case HOST:
          f = (float *)sdt_alloc(t->len * sizeof(float));
          break;
     case DEVICE:
          checkError(cudaMalloc(&f, t->len * sizeof(float)));
          break;
     default:
          fprintf(stderr, "unknown MallocKind %d\n", mkind);
          return NULL;
     }

     t->data = f;
     return t;
}

void freeTensor(Tensor *t, int do_free_data)
{
     assert(isTensorValid(t));
     sdt_free(t->dims);
     if (do_free_data) {
          if (isDeviceMem(t->data))
               checkError(cudaFree(t->data));
          else
               sdt_free(t->data);
     }
     sdt_free(t);
}

void fprintTensor(FILE *stream, const Tensor *tensor, const char *fmt)
{
     assertTensor(tensor);
     int dim_sizes[MAXDIM], dim_levels[MAXDIM]; /* dimision size and how deep current chars go */
     int ndim = tensor->ndim, len = tensor->len, *dims = tensor->dims; /* pointer short cut */
     float *data = tensor->data;
     char left_buf[MAXDIM+1], right_buf[MAXDIM+1]; /* buffer for brackets */
     char *lp = left_buf, *rp = right_buf;
     size_t right_len;
     int i, j, k;

     dim_sizes[ndim-1] = tensor->dims[ndim-1];
     dim_levels[ndim-1] = 0;
     for (i = ndim-2; i >= 0; i--) {
          dim_sizes[i] = dims[i] * dim_sizes[i+1];
          dim_levels[i] = 0;
     }
     for (i = 0; i < len; i++) {
          for (j = 0; j < ndim; j++) {
               if (i % dim_sizes[j] == 0)
                    dim_levels[j]++;
               if (dim_levels[j] == 1) {
                    *lp++ = '[';
                    dim_levels[j]++;
               }
               if (dim_levels[j] == 3) {
                    *rp++ = ']';
                    if (j != 0 && dim_levels[j] > dim_levels[j-1]) {
                         *lp++ = '[';
                         dim_levels[j] = 2;
                    } else
                         dim_levels[j] = 0;
               }
          }
          *lp = *rp = '\0';
          fprintf(stream, "%s", right_buf);
          if (*right_buf != '\0') {
               fprintf(stream, "\n");
               right_len = strlen(right_buf);
               for (k = ndim-right_len; k > 0; k--)
                    fprintf(stream, " ");
          }
          fprintf(stream, "%s", left_buf);
          if (*left_buf == '\0')
               fprintf(stream, " ");
          fprintf(stream, fmt, data[i]);
          lp = left_buf, rp = right_buf;
     }
     for (j = 0; j < ndim; j++)
          fprintf(stream, "]");
     fprintf(stream, "\n");
}

void printTensor(const Tensor *tensor, const char *fmt)
{
     fprintTensor(stdout, tensor, fmt);
}

void fprintDeviceTensor(FILE *stream, const Tensor *d_tensor, const char *fmt)
{
     assert(isTensorValid(d_tensor));
     Tensor *h_tensor = cloneTensor(d_tensor, D2H);
     fprintTensor(stream, h_tensor, fmt);
     free(h_tensor->data); /* TODO: free t_tensor */
}

void printDeviceTensor(const Tensor *d_tensor, const char *fmt)
{
     fprintDeviceTensor(stdout, d_tensor, fmt);
}

void saveTensor(const char *file_name, const Tensor *tensor, const char *fmt)
{
     FILE *fp = fopen(file_name, "w");
     fprintTensor(fp, tensor, fmt);
     fclose(fp);
}

double getUnixTime(void)
{
     struct timespec tv;

     if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

     return (tv.tv_sec + (tv.tv_nsec / 1.0e9));
}

Tensor *createSlicedTensor(const Tensor *src, int dim, int start, int len)
{
     assert(isTensorValid(src));
     assert(dim <= MAXDIM);
     assert(len+start <= src->dims[dim]);

     Tensor *dst = (Tensor *)sdt_alloc(sizeof(Tensor)); /* new tensor */
     dst->ndim = src->ndim;
     dst->dims = (int *)sdt_alloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = len;
     dst->len = src->len / src->dims[dim] * len;
     checkError(cudaMalloc(&dst->data, sizeof(float) * dst->len));
     return dst;
}

__global__ void sliceTensorKernel(float *src_data, float *dst_data, ...)
{
}

Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len)
{
     /* Your code here. You can have a cuda kernel below. */

     /* sliceTensorKernel<<<block_num, block_size>>>(src->data, dst->data, ...) */

     return dst;
}
