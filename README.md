General Information
======================

KIRO is the KITs InfiniBand remote communication library.
It provides a simple server and client class that can be used to pass arbitrary
information from the server to the client using _native_ InfiniBand
communication.
It also provides a network transmittable ring-buffer (KIRO-TRB) which can be used as
a transmission container for same-sized objects and a (uni directional) self-synchronizing buffer (KIRO-SB) which can be used to automatically keep a local object in sync with a shared remote object.

The library is optimized for speed and ease of use.


Installation
=====================

Please refer to the INSTALL file of this project.


Usage
====================

Example KIRO server usage

```C
#include <kiro-server.h>
...

int memSize = 42;
void *mem = malloc(memSize); //This is the memory we want to provide

KiroServer *server = kiro_server_new ();

const char *address = "192.168.1.1";
const char *port = "60010";
kiro_server_start (server, address, port,  mem, memSize);
// The KIRO server is now running

...

kiro_server_stop (server);
kiro_server_free (server);

...
```

Example KIRO client usage
```C
#include <kiro-client.h>
...

KiroClient *client = kiro_client_new ();

const char *address = "192.168.1.1";
const char *port = "60010";

kiro_client_connect (client, address, port);
//The client is now connected

kiro_client_sync (client);

void *mem = kiro_client_get_memory (client);

kiro_client_free (client);
```

For TRB usage, check the examples in the _test_ directory


Licensing
=====================

kiro is copyright to the Karlsruhe Institute of Technology and licensed under
the LGPL 2.1.
