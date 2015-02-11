/**
 * indentity
 *
 * This is a CUDA kernel, that does nothing with the data and is meant for
 * testing purposes.
 *
 **/
__global__
void identity (void *memory_pointer, int memory_size)
{
/*
    int *mem_pointer = (int*) memory_pointer;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    mem_pointer[index/4] *= 2;
*/
}
