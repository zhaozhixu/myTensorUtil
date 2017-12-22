#ifndef _TENSOR_UTIL_H_
#define _TENSOR_UTIL_H_

typedef enum MallocKind {
     HOST, DEVICE
} MallocKind;

typedef enum CloneKind {
     H2H, H2D, D2D, D2H
} CloneKind;

typedef struct {
     int ndim;
     int *dims;
     int len;
     float *data;
} Tensor;

int isTensorValid(const Tensor *tensor);
int isShapeEqual(const Tensor *t1, const Tensor *t2);
int isHostMem(const void *ptr);
int isDeviceMem(const void *ptr);
void *cloneMem(const void *src, size_t size, CloneKind kind);
Tensor *cloneTensor(const Tensor *src, CloneKind kind);
void *repeatMem(void *data, size_t size, int times, CloneKind kind);
int computeLength(int ndim, const int *dims);
Tensor *createTensor(float *data, int ndim, const int *dims);
Tensor *mallocTensor(int ndim, const int* dims, const MallocKind mkind);
void freeTensor(Tensor *t, int do_free_data);

void fprintTensor(FILE *stream, const Tensor *tensor, const char *fmt);
void printTensor(const Tensor *tensor, const char *fmt);
void fprintDeviceTensor(FILE *stream, const Tensor *d_tensor, const char *fmt);
void printDeviceTensor(const Tensor *d_tensor, const char *fmt);
void saveTensor(const char *file_name, const Tensor *tensor, const char *fmt);
void saveDeviceTensor(const char *file_name, const Tensor *d_tensor, const char *fmt);
double getUnixTime(void);

Tensor *createSlicedTensor(const Tensor *src, int dim, int start, int len);
Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len);

#endif  /* _TENSOR_UTIL_H_ */
