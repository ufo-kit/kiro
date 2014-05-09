#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "kiro-client.h"


int main ( int argc, char *argv[] )
{
    if (argc < 3)
    {
        printf("Not enough aruments. Usage: ./client <address> <port>\n");
        return -1;
    }
    KiroClient *client = g_object_new(KIRO_TYPE_CLIENT, NULL);
    if(-1 != kiro_client_connect(client, argv[1], argv[2]))
        kiro_client_sync(client);
    g_object_unref(client);
    return 0;
}