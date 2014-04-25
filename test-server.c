#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "kiro-server.h"


int main(void)
{
    KiroServer *server = g_object_new(KIRO_TYPE_SERVER, NULL);
    kiro_server_start(server, "192.168.11.61", "60010");
    return 0; 
}