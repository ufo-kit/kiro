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
    void *offset_input;
    void *offset_output;
    int i;

    // DEBUG:
    *(int *)(output + sizeof (unsigned long int)) = 42;


    offset_input = input + sizeof (unsigned long int);
    offset_output = output + sizeof (unsigned long int);

    for (i = 0; i < input_size; i += sizeof (int)) {
        *(int *)offset_output = *(int *)offset_input;
        offset_input = offset_input + i;
        offset_output = offset_output + i;
    }

    // Increment frame. This always has to happen last, so the client knows, it's ready.
    *(unsigned long int *)output = *(unsigned long int *)input;

}
