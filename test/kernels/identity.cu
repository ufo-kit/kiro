/**
 * indentity
 *
 * This is a CUDA kernel, that does nothing with the data and is meant for
 * testing purposes.
 *
 **/
__global__
void identity (void *input, int input_size, void *output, int output_size)
{
    // Normal algorithm here.
    int index = (blockIdx.x * blockDim.x + threadIdx.x) + sizeof (long unsigned int);
    ((int *)output)[index/4] = ((int *)input)[index/4];

/*
    void *offset_input;
    void *offset_output;
    int i;

    offset_input = (void *)((int *)input + sizeof (unsigned long int));
    offset_output = (void *)((int *)output + sizeof (unsigned long int));

    for (i = 0; i < input_size - sizeof (unsigned long int); i + sizeof (int)) {
        *(int *)offset_output = *(int *)offset_input;
        offset_input = (void *)((int *)offset_input + sizeof (int));
        offset_output = (void *)((int *)offset_output + sizeof (int));
        
    }
*/
    // Increment frame. This always has to happen last, so the client knows, it's ready.
    *(unsigned long int *)output = *(unsigned long int *)input;

}
