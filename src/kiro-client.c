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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <rdma/rdma_verbs.h>
#include <glib.h>
#include "kiro-client.h"
#include "kiro-rdma.h"
#include "kiro-trb.h"

#include <errno.h>


/*
 * Definition of 'private' structures and members and macro to access them
 */

#define KIRO_CLIENT_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), KIRO_TYPE_CLIENT, KiroClientPrivate))

struct _KiroClientPrivate {

    /* Properties */
    // PLACEHOLDER //

    /* 'Real' private structures */
    /* (Not accessible by properties) */
    struct rdma_event_channel   *ec;        // Main Event Channel
    struct rdma_cm_id           *conn;      // Connection to the Server

};


G_DEFINE_TYPE (KiroClient, kiro_client, G_TYPE_OBJECT);


KiroClient *
kiro_client_new (void)
{
    return g_object_new (KIRO_TYPE_CLIENT, NULL);
}


static void
kiro_client_init (KiroClient *self)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);
    memset (priv, 0, sizeof (&priv));
}

static void
kiro_client_finalize (GObject *object)
{
    //KiroClient *self = KIRO_CLIENT(object);
    //KiroClientPrivate * priv = KIRO_CLIENT_GET_PRIVATE(self);
    //PASS
}

static void
kiro_client_class_init (KiroClientClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    gobject_class->finalize = kiro_client_finalize;
    g_type_class_add_private (klass, sizeof (KiroClientPrivate));
}



int
kiro_client_connect (KiroClient *self, char *address, char *port)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (priv->conn) {
        g_warning ("Already connected to server");
        return -1;
    }

    struct rdma_addrinfo hints, *res_addrinfo;

    memset (&hints, 0, sizeof (hints));

    hints.ai_port_space = RDMA_PS_IB;

    if (rdma_getaddrinfo (address, port, &hints, &res_addrinfo)) {
        g_critical ("Failed to get address information for %s:%s : %s", address, port, strerror (errno));
        return -1;
    }

    g_debug ("Address information created");
    struct ibv_qp_init_attr qp_attr;
    memset (&qp_attr, 0, sizeof (qp_attr));
    qp_attr.cap.max_send_wr = 10;
    qp_attr.cap.max_recv_wr = 10;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.qp_context = priv->conn;
    qp_attr.sq_sig_all = 1;

    if (rdma_create_ep (& (priv->conn), res_addrinfo, NULL, &qp_attr)) {
        g_critical ("Endpoint creation failed: %s", strerror (errno));
        return -1;
    }

    g_debug ("Route to server resolved");
    struct kiro_connection_context *ctx = (struct kiro_connection_context *)calloc (1, sizeof (struct kiro_connection_context));

    if (!ctx) {
        g_critical ("Failed to create connection context (Out of memory?)");
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    ctx->cf_mr_send = (struct kiro_rdma_mem *)calloc (1, sizeof (struct kiro_rdma_mem));
    ctx->cf_mr_recv = (struct kiro_rdma_mem *)calloc (1, sizeof (struct kiro_rdma_mem));

    if (!ctx->cf_mr_recv || !ctx->cf_mr_send) {
        g_critical ("Failed to allocate Control Flow Memory Container (Out of memory?)");
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    ctx->cf_mr_recv = kiro_create_rdma_memory (priv->conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);
    ctx->cf_mr_send = kiro_create_rdma_memory (priv->conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);

    if (!ctx->cf_mr_recv || !ctx->cf_mr_send) {
        g_critical ("Failed to register control message memory (Out of memory?)");
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    ctx->cf_mr_recv->size = ctx->cf_mr_send->size = sizeof (struct kiro_ctrl_msg);
    priv->conn->context = ctx;

    if (rdma_post_recv (priv->conn, priv->conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        g_critical ("Posting preemtive receive for connection failed: %s", strerror (errno));
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    if (rdma_connect (priv->conn, NULL)) {
        g_critical ("Failed to establish connection to the server: %s", strerror (errno));
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    g_message ("Connection to server established");
    struct ibv_wc wc;

    if (rdma_get_recv_comp (priv->conn, &wc) < 0) {
        g_critical ("Failure waiting for POST from server: %s", strerror (errno));
        rdma_disconnect (priv->conn);
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    g_debug ("Got RDMI Access information from Server");
    ctx->peer_mr = (((struct kiro_ctrl_msg *) (ctx->cf_mr_recv->mem))->peer_mri);
    g_debug ("Expected Memory Size is: %u", ctx->peer_mr.length);
    ctx->rdma_mr = kiro_create_rdma_memory (priv->conn->pd, ctx->peer_mr.length, IBV_ACCESS_LOCAL_WRITE);

    if (!ctx->rdma_mr) {
        g_critical ("Failed to allocate memory for receive buffer (Out of memory?)");
        rdma_disconnect (priv->conn);
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    g_message ("Connected to %s:%s", address, port);
    return 0;
}



int
kiro_client_sync (KiroClient *self)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);
    struct kiro_connection_context *ctx = (struct kiro_connection_context *)priv->conn->context;

    if (rdma_post_read (priv->conn, priv->conn, ctx->rdma_mr->mem, ctx->peer_mr.length, ctx->rdma_mr->mr, 0, ctx->peer_mr.addr, ctx->peer_mr.rkey)) {
        g_critical ("Failed to RDMA_READ from server: %s", strerror (errno));
        rdma_disconnect (priv->conn);
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    struct ibv_wc wc;

    if (rdma_get_send_comp (priv->conn, &wc) < 0) {
        g_critical ("No send completion for RDMA_READ received: %s", strerror (errno));
        rdma_disconnect (priv->conn);
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    return 0;
}


void *
kiro_client_get_memory (KiroClient *self)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (!priv->conn)
        return NULL;

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)priv->conn->context;

    if (!ctx->rdma_mr)
        return NULL;

    return ctx->rdma_mr->mem;
}


size_t 
kiro_client_get_memory_size (KiroClient *self)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (!priv->conn)
        return 0;

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)priv->conn->context;

    if (!ctx->rdma_mr)
        return 0;

    return ctx->rdma_mr->size;
}





