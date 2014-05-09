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
 * SECTION: kiro-client
 * @Short_description: KIRO RDMA Client / Consumer
 * @Title: KiroClient
 *
 * KiroClient implements the client / active / consumer side of the the RDMA
 * Communication Channel. It uses a KIRO-CLIENT to manage data read from the Server.
 */
 
#ifndef __KIRO_CLIENT_H
#define __KIRO_CLIENT_H

#include <stdint.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define KIRO_TYPE_CLIENT             (kiro_client_get_type())
#define KIRO_CLIENT(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), KIRO_TYPE_CLIENT, KiroClient))
#define KIRO_IS_CLIENT(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), KIRO_TYPE_CLIENT))
#define KIRO_CLIENT_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), KIRO_TYPE_CLIENT, KiroClientClass))
#define KIRO_IS_CLIENT_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), KIRO_TYPE_CLIENT))
#define KIRO_CLIENT_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj), KIRO_TYPE_CLIENT, KiroClientClass))


typedef struct _KiroClient           KiroClient;
typedef struct _KiroClientClass      KiroClientClass;
typedef struct _KiroClientPrivate    KiroClientPrivate;


struct _KiroClient {
    
    GObject parent;
    
    /*< private >*/
    KiroClientPrivate *priv;
};


/**
 * IbvConnectorInterface:
 *
 * Base interface for IbvConnectors.
 */

struct _KiroClientClass {
    
    GObjectClass parent_class;
       
};



/* GObject and GType functions */
GType       kiro_client_get_type            (void);

GObject     kiro_client_new                 (void);

/* client functions */


int         kiro_client_connect             (KiroClient* client, char* dest_addr, char* dest_port);

int         kiro_client_sync                (KiroClient* client);

void*       kiro_client_get_memory          (KiroClient* client);

size_t      kior_client_get_memory_size     (KiroClient* client);

G_END_DECLS

#endif //__KIRO_CLIENT_H