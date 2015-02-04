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
    guint                       rdma_ec_id;   // ID of the GSource that will be created for the rdma_ec GIOChannel
    guint                       conn_ec_id;   // ID of the GSource that will be created for the conn_ec GIOChannel
    GThread                     *main_thread; // Main KIRO client thread
};


G_DEFINE_TYPE (KiroClient, kiro_client, G_TYPE_OBJECT);

// Temporary storage and lock for PING timing
G_LOCK_DEFINE (ping_time);
volatile struct timeval ping_time;

G_LOCK_DEFINE (sync_lock);

static inline gboolean
send_msg (struct rdma_cm_id *id, struct kiro_rdma_mem *r)
{
    gboolean retval = TRUE;
    G_LOCK (sync_lock);
    if (rdma_post_send (id, id, r->mem, r->size, r->mr, IBV_SEND_SIGNALED)) {
        retval = FALSE;
    }
    else {
        struct ibv_wc wc;
        if (rdma_get_send_comp (id, &wc) < 0) {
            retval = FALSE;
        }
        g_debug ("WC Status: %i", wc.status);
    }

    G_UNLOCK (sync_lock);
    return retval;
}


KiroClient *
kiro_client_new (void)
{
    return g_object_new (KIRO_TYPE_CLIENT, NULL);
}


void
kiro_client_free (KiroClient *client)
{
    g_return_if_fail (client != NULL);

    if (KIRO_IS_CLIENT (client))
        g_object_unref (client);
    else
        g_warning ("Trying to use kiro_client_free on an object which is not a KIRO client. Ignoring...");
}


static void
kiro_client_init (KiroClient *self)
{
    g_return_if_fail (self != NULL);
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);
    memset (priv, 0, sizeof (&priv));
    //Hack to make the 'unused function' from the kiro-rdma include go away...
    kiro_attach_qp (NULL);
    ping_time.tv_sec = -1;
    ping_time.tv_usec = -1;
}


static void
kiro_client_finalize (GObject *object)
{
    g_return_if_fail (object != NULL);
    if (KIRO_IS_CLIENT (object))
        kiro_client_disconnect ((KiroClient *)object);
    G_OBJECT_CLASS (kiro_client_parent_class)->finalize (object);
}


static void
kiro_client_class_init (KiroClientClass *klass)
{
    g_return_if_fail (klass != NULL);
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
        struct rdma_cm_event *ev = g_try_malloc (sizeof (*active_event));

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



static gboolean
process_rdma_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    //(void) condition;
    g_debug ("Message condidition: %i", condition);

    KiroClientPrivate *priv = (KiroClientPrivate *)data;
    struct ibv_wc wc;

    if (ibv_poll_cq (priv->conn->recv_cq, 1, &wc) < 0) {
        g_critical ("Failure getting receive completion event from the queue: %s", strerror (errno));
        return FALSE;
    }
    void *cq_ctx;
    struct ibv_cq *cq;
    int err = ibv_get_cq_event (priv->conn->recv_cq_channel, &cq, &cq_ctx);
    if (!err)
        ibv_ack_cq_events (cq, 1);

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)priv->conn->context;
    guint type = ((struct kiro_ctrl_msg *)ctx->cf_mr_recv->mem)->msg_type;
    g_debug ("Received a message from the Server of type: %u", type);

    if (type == KIRO_ACK_RDMA) {
        g_debug ("Got RDMI Access information from Server");
        ctx->peer_mr = (((struct kiro_ctrl_msg *) (ctx->cf_mr_recv->mem))->peer_mri);
        g_debug ("Expected Memory Size is: %zu", ctx->peer_mr.length);
        ctx->rdma_mr = kiro_create_rdma_memory (priv->conn->pd, ctx->peer_mr.length, IBV_ACCESS_LOCAL_WRITE);

        if (!ctx->rdma_mr) {
            //FIXME: Connection teardown in an event handler routine? Not a good
            //idea...
            g_critical ("Failed to allocate memory for receive buffer (Out of memory?)");
            rdma_disconnect (priv->conn);
            kiro_destroy_connection_context (&ctx);
            rdma_destroy_ep (priv->conn);
            return TRUE;
        }
    }
    if (type == KIRO_PONG) {
        G_LOCK (ping_time);
        struct timeval local_time;
        gettimeofday (&local_time, NULL);

        if (ping_time.tv_sec == 0 && ping_time.tv_usec == 0) {
            g_debug ("Received PONG message from server");
            ping_time.tv_sec = local_time.tv_sec;
            ping_time.tv_usec = local_time.tv_usec;
        }
        else {
            g_debug ("Received unexpected PONG message from server");
        }

        G_UNLOCK (ping_time);
    }
    if (type == KIRO_REALLOC) {
        g_debug ("Got reallocation request from server.");
        struct kiro_ctrl_msg *msg = ((struct kiro_ctrl_msg *)ctx->cf_mr_recv->mem);

        G_LOCK (sync_lock);
        g_debug ("Rallocating memory...");
        kiro_destroy_rdma_memory (ctx->rdma_mr);
        ctx->peer_mr = msg->peer_mri;
        g_debug ("New size is: %zu", ctx->peer_mr.length);
        ctx->rdma_mr = kiro_create_rdma_memory (priv->conn->pd, ctx->peer_mr.length, IBV_ACCESS_LOCAL_WRITE);
        G_UNLOCK (sync_lock);

        if (!ctx->rdma_mr) {
            //FIXME: Connection teardown in an event handler routine? Not a good
            //idea...
            g_critical ("Failed to allocate memory for receive buffer (Out of memory?)");
            rdma_disconnect (priv->conn);
            kiro_destroy_connection_context (&ctx);
            rdma_destroy_ep (priv->conn);
        }

        msg = ((struct kiro_ctrl_msg *)ctx->cf_mr_send->mem);
        msg->msg_type = KIRO_ACK_RDMA;
        if (!send_msg (priv->conn, ctx->cf_mr_send)) {
            g_warning ("Failure while trying to post SEND for reallocation ACK: %s", strerror (errno));
        }
        else {
            g_debug ("Sent ACK to server");
        }
    }

    //Post a generic receive in order to stay responsive to any messages from
    //the server
    if (rdma_post_recv (priv->conn, priv->conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        //FIXME: Connection teardown in an event handler routine? Not a good
        //idea...
        g_critical ("Posting generic receive for connection failed: %s", strerror (errno));
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        return FALSE;
    }

    // make sure the next incoming work completion causes an event on the
    // receive completion channel. We will poll() the channels file descriptor
    // for this in the kiro client main loop.
    ibv_req_notify_cq (priv->conn->recv_cq, 0);

    g_debug ("Finished RDMA event handling");
    return TRUE;
}


gpointer
start_client_main_loop (gpointer data)
{
    g_main_loop_run ((GMainLoop *)data);
    return NULL;
}


gboolean
stop_client_main_loop (KiroClientPrivate *priv)
{
    if (priv->close_signal) {
        // Get the IO Channels and destroy them.
        // This will also unref their respective GIOChannels
        GSource *tmp = g_main_context_find_source_by_id (NULL, priv->conn_ec_id);
        g_source_destroy (tmp);
        priv->conn_ec_id = 0;

        tmp = g_main_context_find_source_by_id (NULL, priv->rdma_ec_id);
        g_source_destroy (tmp);
        priv->rdma_ec_id = 0;

        g_main_loop_quit (priv->main_loop);
        g_debug ("Event handling stopped");
        return FALSE;
    }
    return TRUE;
}


int
kiro_client_connect (KiroClient *self, const char *address, const char *port)
{
    g_return_val_if_fail (self != NULL, -1);
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
    struct kiro_connection_context *ctx = (struct kiro_connection_context *)g_try_malloc0 (sizeof (struct kiro_connection_context));

    if (!ctx) {
        g_critical ("Failed to create connection context (Out of memory?)");
        rdma_destroy_ep (priv->conn);
        return -1;
    }

    ctx->cf_mr_recv = kiro_create_rdma_memory (priv->conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);
    ctx->cf_mr_send = kiro_create_rdma_memory (priv->conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);

    if (!ctx->cf_mr_recv || !ctx->cf_mr_send) {
        g_critical ("Failed to register control message memory (Out of memory?)");
        goto fail;
    }

    ctx->cf_mr_recv->size = ctx->cf_mr_send->size = sizeof (struct kiro_ctrl_msg);
    priv->conn->context = ctx;

    //Post an preemtive receive for the servers welcome message
    if (rdma_post_recv (priv->conn, priv->conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        g_critical ("Posting preemtive receive for connection failed: %s", strerror (errno));
        goto fail;
    }

    if (rdma_connect (priv->conn, NULL)) {
        g_critical ("Failed to establish connection to the server: %s", strerror (errno));
        goto fail;
    }

    g_message ("Connection to server established. Waiting for response.");
    ibv_req_notify_cq (priv->conn->recv_cq, 0); // Make the respective Queue push events onto the channel
    if (!process_rdma_event (NULL, 0, (gpointer)priv)) {
        g_critical ("No RDMA access information received from the server. Failed to connect.");
        goto fail;
    }

    g_message ("Connected to %s:%s", address, port);

    priv->ec = priv->conn->channel; //For easy access
    priv->main_loop = g_main_loop_new (NULL, FALSE);
    g_idle_add ((GSourceFunc)stop_client_main_loop, priv);
    GIOChannel *conn_ec = g_io_channel_unix_new (priv->ec->fd);
    priv->conn_ec_id = g_io_add_watch (conn_ec, G_IO_IN | G_IO_PRI, process_cm_event, (gpointer)priv);
    GIOChannel *rdma_ec = g_io_channel_unix_new (priv->conn->recv_cq_channel->fd);
    priv->rdma_ec_id = g_io_add_watch (rdma_ec, G_IO_IN | G_IO_PRI, process_rdma_event, (gpointer)priv);
    priv->main_thread = g_thread_new ("KIRO Client main loop", start_client_main_loop, priv->main_loop);

    // We gave control to the main_loop (with add_watch) and don't need our ref
    // any longer
    g_io_channel_unref (conn_ec);
    g_io_channel_unref (rdma_ec);

    return 0;

fail:
    kiro_destroy_connection_context (&ctx);
    rdma_destroy_ep (priv->conn);
    priv->conn = NULL;
    return -1;
}


int
kiro_client_sync_partial (KiroClient *self, gulong remote_offset, gulong size, gulong local_offset)
{
    g_return_val_if_fail (self != NULL, -1);
    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (!priv->conn) {
        g_warning ("Client not connected");
        return -1;
    }

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)priv->conn->context;

    if (remote_offset > ctx->peer_mr.length) {
        g_warning ("kiro_client_sync_partial: remote_offset too large! Won't sync.");
        return -1;
    }

    gulong read_size = ctx->peer_mr.length;
    if (size > 0)
        read_size = size;
    else if (remote_offset > 0)
        read_size -= remote_offset;  //read to the end of the memory, starting at offset

    if ((remote_offset + read_size) > ctx->peer_mr.length) {
        g_warning ("kiro_client_sync_partial: remote_offset + read_size would exceed remote memory boundary! Won't sync.");
        return -1;
    }

    if ((local_offset + read_size) > ctx->rdma_mr->size) {
        g_warning ("kiro_client_sync_partial: local_offset + read_size would exceed local memory boundary! Won't sync.");
        return -1;
    }

    G_LOCK (sync_lock);
    if (rdma_post_read (priv->conn, priv->conn, ctx->rdma_mr->mem + local_offset, read_size, ctx->rdma_mr->mr, 0, (uint64_t)ctx->peer_mr.addr + remote_offset, ctx->peer_mr.rkey)) {
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
            G_UNLOCK (sync_lock);
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
    G_UNLOCK (sync_lock);
    return -1;
}


int
kiro_client_sync (KiroClient *self)
{
    return kiro_client_sync_partial (self, 0, 0, 0);
}


gboolean
ping_timeout (gpointer data) {

    //Not needed. Void it to prevent 'unused variable' warning
    (void) data;
    g_debug ("PING timed out");

    G_LOCK (ping_time);

    // Maybe the server did answer while dispatching the timeout?
    if (ping_time.tv_sec != 0 || ping_time.tv_usec != 0) {
        goto done;
    }

    ping_time.tv_usec = -1;
    ping_time.tv_sec = -1;


done:
    G_UNLOCK (ping_time);

    // Return FALSE to automtically stop the timeout from reoccuring
    return FALSE;
}


gint
kiro_client_ping_server (KiroClient *self)
{
    g_return_val_if_fail (self != NULL, -1);

    // Will be returned. -1 for error.
    gint t_usec = 0;

    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);
    if (!priv->conn) {
        g_warning ("Client not connected");
        return -1;
    }

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)priv->conn->context;

    struct kiro_ctrl_msg *msg = (struct kiro_ctrl_msg *)(ctx->cf_mr_send->mem);
    msg->msg_type = KIRO_PING;

    G_LOCK (ping_time);
    ping_time.tv_sec = 0;
    ping_time.tv_usec = 0;
    struct timeval local_time;
    gettimeofday (&local_time, NULL);

    if (!send_msg (priv->conn, ctx->cf_mr_send)) {
        g_warning ("Failure while trying to post SEND for PING: %s", strerror (errno));
        t_usec = -1;
        G_UNLOCK (ping_time);
        goto end;
    }
    g_debug ("PING message sent to server.");
    G_UNLOCK (ping_time);

    // Set a two-second timeout for the ping
    guint timeout = g_timeout_add_seconds (2, ping_timeout, NULL);

    //Wait for ping response
    while (ping_time.tv_sec == 0 && ping_time.tv_usec == 0) {};


    G_LOCK (ping_time);
    // No response from the server. Timeout kicked in
    // (Note: The timeout callback has already deregistered itself. We don't
    // need to do that here again)
    if (ping_time.tv_sec == -1 && ping_time.tv_usec == -1) {
        g_message ("PING timed out.");
        G_UNLOCK (ping_time);
        t_usec = -1;
        goto end;
    }

    // Remove the timeout
    GSource *timeout_source = g_main_context_find_source_by_id (NULL, timeout);
    if (timeout_source) {
        g_source_destroy (timeout_source);
    }

    gint secs = ping_time.tv_sec - local_time.tv_sec;

    // tv_usecs wraps back to 0 at 1000000us (1s).
    // This might cause our calculation to produce negative numbers when time > 1s.
    for (int i = 0; i < secs; i++) {
        ping_time.tv_usec += 1000 * 1000;
    }
    t_usec = ping_time.tv_usec - local_time.tv_usec;
    gint millis = (gint)(t_usec/1000.);
    G_UNLOCK (ping_time);

    g_debug ("Server responded to PING in: %is, %ims, %ius", secs, millis, t_usec);

end:
    G_LOCK (ping_time);
    ping_time.tv_sec = -1;
    ping_time.tv_usec = -1;
    G_UNLOCK (ping_time);
    return t_usec;
}


void *
kiro_client_get_memory (KiroClient *self)
{
    g_return_val_if_fail (self != NULL, NULL);
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
    g_return_val_if_fail (self != NULL, 0);
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
    g_return_if_fail (self != NULL);

    KiroClientPrivate *priv = KIRO_CLIENT_GET_PRIVATE (self);

    if (!priv->conn)
        return;

    //Shut down event listening
    priv->close_signal = TRUE;
    while (g_main_loop_is_running (priv->main_loop)) {};

    // Main loop stopped. Clear its memory
    g_main_loop_unref (priv->main_loop);
    priv->main_loop = NULL;

    // Ask the main thread to join (It probably already has, but we do it
    // anyways. Just in case!)
    g_thread_join (priv->main_thread);
    g_thread_unref (priv->main_thread);
    priv->main_thread = NULL;

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

