/* Copyright (C) 2014 Timo Dritschler <timo.dritschler@kit.edu>
   (Karlsruhe Institute of Technology)

   This library is free software; you can redistribute it and/or modify it
   under the terms of the GNU Lesser General Public License as published by the
   Free Software Foundation; either version 2.1 of the License, or (at your
   option) any later version.

   This library is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
   details.

   You should have received a copy of the GNU Lesser General Public License along
   with this library; if not, write to the Free Software Foundation, Inc., 51
   Franklin St, Fifth Floor, Boston, MA 02110, USA
*/

/**
 * SECTION: kiro-server
 * @Short_description: KIRO RDMA Server / Consumer
 * @Title: KiroServer
 *
 * KiroServer implements the server / passive / provider side of the the RDMA
 * Communication Channel. It uses a KIRO-TRB to manage its data.
 */
 
#ifndef __KIRO_SERVER_H
#define __KIRO_SERVER_H

#include <stdint.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define KIRO_TYPE_SERVER             (kiro_server_get_type())
#define KIRO_SERVER(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), KIRO_TYPE_SERVER, KiroServer))
#define KIRO_IS_SERVER(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), KIRO_TYPE_SERVER))
#define KIRO_SERVER_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), KIRO_TYPE_SERVER, KiroServerClass))
#define KIRO_IS_SERVER_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), KIRO_TYPE_SERVER))
#define KIRO_SERVER_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj), KIRO_TYPE_SERVER, KiroServerClass))


typedef struct _KiroServer           KiroServer;
typedef struct _KiroServerClass      KiroServerClass;
typedef struct _KiroServerPrivate    KiroServerPrivate;


struct _KiroServer {
    
    GObject parent;
    
    /*< private >*/
    KiroServerPrivate *priv;
};


/**
 * IbvConnectorInterface:
 *
 * Base interface for IbvConnectors.
 */

struct _KiroServerClass {
    
    GObjectClass parent_class;
       
};



/* GObject and GType functions */
GType        kiro_server_get_type            (void);

KiroServer*  kiro_server_new                 (void);

/* server functions */

/**
 * kiro_server_start - Starts the server, providing the given memory
 * @server: #KiroServer to perform the operation on
 * @bind_addr: Local address to bind the server to
 * @bind_port: Local port to listen for connections
 * @mem: Pointer to the memory that is to be provided
 * @mem_size: Size in bytes of the given memory
 * Description:
 *   Starts the #KiroServer to provide the given memory to any connecting
 *   client.
 * Notes:
 *   If the bind_addr is NULL, the server will bind to the first device
 *   it can find on the machine and listen across all IPs. Otherwise it
 *   will try to bind to the device associated with the given address.
 *   Address is given as a string of either a hostname or a dot-seperated
 *   IPv4 address or a colon-seperated IPv6 hex-address.
 *   If bind_port is NULL the server will choose a free port randomly
 *   and return the chosen port as return value.
 *   If server creation fails, -1 is returned instead.
 * See also:
 *   kiro_trb_reshape, kiro_trb_adopt,
 *   kiro_trb_clone
 */
int kiro_server_start (KiroServer* server, char* bind_addr, char* bind_port, void* mem, size_t mem_size);

/**
 * kiro_server_stop - Stops the server
 * @server: #KiroServer to perform the operation on
 * Description:
 *   Stops the given #KiroServer
 * See also:
 *   kiro_server_start
 */
void kiro_server_stop (KiroServer* server);

G_END_DECLS

#endif //__KIRO_SERVER_H