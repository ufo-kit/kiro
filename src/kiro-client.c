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
    struct rdma_event_channel   *ec;          // Main Event Channel
    struct rdma_cm_id           *conn;        // Connection to the Server

    gboolean                    close_signal; // Flag used to signal event listening to stop for connection tear-down
    GMainLoop                   *main_loop;   // Main loop of the server for event polling and handling
    GIOChannel                  *g_io_ec;     // GLib IO Channel encapsulation for the connection manager event channel
    GThread                     *main_thread; // Main KIRO client thread
};


G_DEFINE_TYPE (KiroClient, kiro_client, G_TYPE_OBJECT);


KiroClient *
kiro_client_new (void)
{
    return g_object_new (KIRO_TYPE_CLIENT, NULL);
}


void
kiro_client_free (KiroClient *client)
{
    if (!client)
        return;

    if (KIRO_IS_CLIENT (client))
        g_object_unref (client);
    else
        g_warning ("Trying to use kiro_client_free on an object which is not a KIRO client. Ignoring...");
}


static void
kiro_client_init (KiroClient *self)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);
    memset (priv, 0, sizeof (&priv));

    //Hack to make the 'unused function' from the kiro-rdma include go away...
    kiro_attach_qp (NULL);
}


static void
kiro_client_finalize (GObject *object)
{
    if (KIRO_IS_CLIENT (object))
        kiro_client_disconnect ((KiroClient *)object);
    G_OBJECT_CLASS (kiro_client_parent_class)->finalize (object);
}


static void
kiro_client_class_init (KiroClientClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    gobject_class->finalize = kiro_client_finalize;
    g_type_class_add_private (klass, sizeof (KiroClientPrivate));
}


static gboolean
process_cm_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    (void) condition;

    KiroClientPrivate *priv = (KiroClientPrivate *)data;
    struct rdma_cm_event *active_event;

    if (0 <= rdma_get_cm_event (priv->ec, &active_event)) {
        //Disable cancellation to prevent undefined states during shutdown
        struct rdma_cm_event *ev = malloc (sizeof (*active_event));

        if (!ev) {
            g_critical ("Unable to allocate memory for Event handling!");
            rdma_ack_cm_event (active_event);
            return FALSE;
        }

        memcpy (ev, active_event, sizeof (*active_event));
        rdma_ack_cm_event (active_event);

        if (ev->event == RDMA_CM_EVENT_DISCONNECTED) {
            g_debug ("Connection closed by server");
        }

        free (ev);
    }
    return TRUE;
}


gpointer
start_client_main_loop (gpointer data)
{
    g_main_loop_run ((GMainLoop *)data);
    return NULL;
}


int
kiro_client_connect (KiroClient *self, const char *address, const char *port)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (priv->conn) {
        g_warning ("Already connected to server");
        return -1;
    }

    struct rdma_addrinfo hints, *res_addrinfo;

    memset (&hints, 0, sizeof (hints));
    hints.ai_port_space = RDMA_PS_IB;

    char *addr_c = g_strdup (address);
    char *port_c = g_strdup (port);
    int rtn = rdma_getaddrinfo (addr_c, port_c, &hints, &res_addrinfo);
    g_free (addr_c);
    g_free (port_c);

    if (rtn) {
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
    priv->ec = priv->conn->channel; //For easy access
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
    g_debug ("Expected Memory Size is: %zu", ctx->peer_mr.length);
    ctx->rdma_mr = kiro_create_rdma_memory (priv->conn->pd, ctx->peer_mr.length, IBV_ACCESS_LOCAL_WRITE);

    if (!ctx->rdma_mr) {
        g_critical ("Failed to allocate memory for receive buffer (Out of memory?)");
        rdma_disconnect (priv->conn);
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    priv->main_loop = g_main_loop_new (NULL, FALSE);
    priv->g_io_ec = g_io_channel_unix_new (priv->ec->fd);
    g_io_add_watch (priv->g_io_ec, G_IO_IN | G_IO_PRI | G_IO_ERR | G_IO_HUP, process_cm_event, (gpointer)priv);
    priv->main_thread = g_thread_new ("KIRO Client main loop", start_client_main_loop, priv->main_loop);

    // We gave control to the main_loop (with add_watch) and don't need our ref
    // any longer
    g_io_channel_unref (priv->g_io_ec);

    g_message ("Connected to %s:%s", address, port);
    return 0;
}


int
kiro_client_sync (KiroClient *self)
{
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (!priv->conn) {
        g_warning ("Client not connected");
        return -1;
    }

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)priv->conn->context;

    if (rdma_post_read (priv->conn, priv->conn, ctx->rdma_mr->mem, ctx->peer_mr.length, ctx->rdma_mr->mr, 0, (uint64_t)ctx->peer_mr.addr, ctx->peer_mr.rkey)) {
        g_critical ("Failed to RDMA_READ from server: %s", strerror (errno));
        goto fail;
    }

    struct ibv_wc wc;

    if (rdma_get_send_comp (priv->conn, &wc) < 0) {
        g_critical ("No send completion for RDMA_READ received: %s", strerror (errno));
        goto fail;
    }

    switch (wc.status) {
        case IBV_WC_SUCCESS:
            return 0;
        case IBV_WC_RETRY_EXC_ERR:
            g_critical ("Server no longer responding");
            break;
        case IBV_WC_REM_ACCESS_ERR:
            g_critical ("Server has revoked access right to read data");
            break;
        default:
            g_critical ("Could not get data from server. Status %u", wc.status);
    }

fail:
    kiro_destroy_connection (&(priv->conn)); 
    return -1;
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


void
kiro_client_disconnect (KiroClient *self)
{
    if (!self)
        return;

    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (!priv->conn)
        return;

    //Shut down event listening
    priv->close_signal = TRUE;
    g_debug ("Event handling stopped");

    // Stop the main loop and clear its memory
    g_main_loop_quit (priv->main_loop);
    g_main_loop_unref (priv->main_loop);
    priv->main_loop = NULL;

    // Ask the main thread to join (It probably already has, but we do it
    // anyways. Just in case!)
    g_thread_join (priv->main_thread);
    priv->main_thread = NULL;

    // We don't need the connection management IO channel container any more.
    // Unref and thus free it.
    g_io_channel_unref (priv->g_io_ec);
    priv->g_io_ec = NULL;

    priv->close_signal = FALSE;

    //kiro_destroy_connection does not free RDMA memory. Therefore, we need to
    //cache the memory pointer and free the memory afterwards manually 
    struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->conn->context);
    void *rdma_mem = ctx->rdma_mr->mem;
    kiro_destroy_connection (&(priv->conn));
    free (rdma_mem);

    // priv->ec is just an easy-access pointer. Don't free it. Just NULL it 
    priv->ec = NULL;
    g_message ("Client disconnected from server");
}

