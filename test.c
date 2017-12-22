#include <time.h>
#include <stdio.h>
#include <string.h>
#include "tensorUtil.h"

double start, end;
Tensor *t_host, *t_device;

int ndim = 4;
int dims[] = {1, 3, 2, 3};
float data[] = {0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0,
                16.0, 17.0};
void init()
{
     t_host = createTensor(data, ndim, dims);
     t_device = cloneTensor(t_host, H2D);
     printf("Original tensor:\n");
     printTensor(t_host, "%.2f");
}

void testSliceTensor()
{
     Tensor *t_sliced_device = createSlicedTensor(t_device, 1, 0, 2);
     start = getUnixTime();
     sliceTensor(t_device, t_sliced_device, 1, 0, 2);
     end = getUnixTime();
     printf("sliceTensor in %fms:\n", end - start);
     printDeviceTensor(t_sliced_device, "%.2f");
}

int main(int argc, char *argv[])
{
     init();
     testSliceTensor();
}
