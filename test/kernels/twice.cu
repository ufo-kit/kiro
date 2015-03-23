/**
 * double
 *
 * This is a very simple CUDA kernel that doubles the integers from one gpu memory region
 * and writes the new integers to another gpu memory region.
 *
 **/
__global__
void twice (void *input, int input_size, void *output, int output_size)
{
    // Normal algorithm here.
    int index = (blockIdx.x * blockDim.x + threadIdx.x) + sizeof (long unsigned int);
    ((int *)output)[index/4] = 2*((int *)input)[index/4];

    // Increment frame. This always has to happen last, so the client knows, it's ready.
    *(unsigned long int *)output = *(unsigned long int *)input;

}
