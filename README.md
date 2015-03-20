General Information
======================

KIRO is the KITs InfiniBand remote communication library.
It provides a simple server and client class that can be used to pass arbitrary
information from the server to the client using _native_ InfiniBand
communication.
It also provides a network transmittable ring-buffer (TRB) which can be used as
a transmission container for same-sized objects.

The library is optimized for speed and ease of use.


Installation
=====================

Please refer to the INSTALL file of this project.


Usage
====================

Example KIRO server usage

```
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
```
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


GPUdirect
=====================


To compile with GPUdirect run cmake with -DGPUDIRECT=ON flag. Example usage is shown in test/test-client-gpudirect.cu and test/test-server-gpudirect.c.


Tested with:
----------
* NVIDIA Tesla K40c
* Mellanox ConnectX-3 MT27500
* Supermicro X10SRi-F
* Intel® C612 chipset
* Intel® Xeon® CPU E5-1630 v3 @ 3.70GHz
* Fedora 21
* Kernel 3.17.4-301.fc21.x86_64
* Cuda 6.5  ([https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads))
* nvidia driver 340.29 (comes with cuda)
* mlnx\_ofed ([http://www.mellanox.com/page/products_dyn?product_family=26](http://www.mellanox.com/page/products_dyn?product_family=26))
* nv\_peer\_mem ([http://www.mellanox.com/page/products_dyn?product_family=116](http://www.mellanox.com/page/products_dyn?product_family=116))

Issues:
----------

* Sometimes the nvidia driver crashes during boot. 
    * Solution: grep for "Oops" in dmesg and reboot if that happens.
* Sometimes mlx4\_core driver is not assigned to the mellanox card.
    * Solution: Find out device number of mellanox card with lspci (e.g. 3) and run:


---        
    echo "1" > /sys/bus/pci/devices/0000\:03\:00.0/remove
    echo "1" > /sys/bus/pci/rescan
---


Licensing
=====================

kiro is copyright to the Karlsruhe Institute of Technology and licensed under
the LGPL 2.1.
