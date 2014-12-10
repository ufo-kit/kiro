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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <rdma/rdma_verbs.h>
#include <glib.h>
#include "kiro-server.h"
#include "kiro-rdma.h"
#include "kiro-trb.h"


/*
 * Definition of 'private' structures and members and macro to access them
 */

#define KIRO_SERVER_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), KIRO_TYPE_SERVER, KiroServerPrivate))

struct _KiroServerPrivate {

    /* Properties */
    // PLACEHOLDER //

    /* 'Real' private structures */
    /* (Not accessible by properties) */
    struct rdma_event_channel   *ec;             // Main Event Channel
    struct rdma_cm_id           *base;           // Base-Listening-Connection
    GList                       *clients;        // List of connected clients
    guint                       next_client_id;  // Numeric ID for the next client that will connect
    void                        *mem;            // Pointer to the server buffer
    size_t                      mem_size;        // Server Buffer Size in bytes

    gboolean                    close_signal;    // Flag used to signal event listening to stop for server shutdown
    GMainLoop                   *main_loop;      // Main loop of the server for event polling and handling
    GIOChannel                  *conn_ec;        // GLib IO Channel encapsulation for the connection manager event channel
    GThread                     *main_thread;    // Main KIRO server thread
};


G_DEFINE_TYPE (KiroServer, kiro_server, G_TYPE_OBJECT);


struct kiro_client_connection {

    guint                       id;              // Client identification (Easy access)
    GIOChannel                  *rcv_ec;         // GLib IO Channel encapsulation for receive completions for the client
    guint                       source_id;       // ID of the source created by g_io_add_watch, needed to remove it again
    struct rdma_cm_id           *conn;           // Connection Manager ID of the client
};


KiroServer *
kiro_server_new (void)
{
    return g_object_new (KIRO_TYPE_SERVER, NULL);
}


void
kiro_server_free (KiroServer *server)
{
    if (!server)
        return;

    if (KIRO_IS_SERVER (server))
        g_object_unref (server);
    else
        g_warning ("Trying to use kiro_server_free on an object which is not a KIRO server. Ignoring...");
}


static void
kiro_server_init (KiroServer *self)
{
    KiroServerPrivate *priv = KIRO_SERVER_GET_PRIVATE (self);
    memset (priv, 0, sizeof (&priv));
}


static void
kiro_server_finalize (GObject *object)
{
    KiroServer *self = KIRO_SERVER (object);
    //Clean up the server
    kiro_server_stop (self);

    G_OBJECT_CLASS (kiro_server_parent_class)->finalize (object);
}


static void
kiro_server_class_init (KiroServerClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    gobject_class->finalize = kiro_server_finalize;
    g_type_class_add_private (klass, sizeof (KiroServerPrivate));
}


static int
connect_client (struct rdma_cm_id *client)
{
    if (!client)
        return -1;

    if ( -1 == kiro_attach_qp (client)) {
        g_critical ("Could not create a QP for the new connection");
        rdma_destroy_id (client);
        return -1;
    }

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)g_try_malloc0 (sizeof (struct kiro_connection_context));

    if (!ctx) {
        g_critical ("Failed to create connection context");
        rdma_destroy_id (client);
        return -1;
    }

    ctx->cf_mr_recv = kiro_create_rdma_memory (client->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);
    ctx->cf_mr_send = kiro_create_rdma_memory (client->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);

    if (!ctx->cf_mr_recv || !ctx->cf_mr_send) {
        g_critical ("Failed to register control message memory");
        goto error;
    }

    ctx->cf_mr_recv->size = ctx->cf_mr_send->size = sizeof (struct kiro_ctrl_msg);
    client->context = ctx;

    if (rdma_post_recv (client, client, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        g_critical ("Posting preemtive receive for connection failed: %s", strerror (errno));
        goto error;
    }

    if (rdma_accept (client, NULL)) {
        g_warning ("Failed to establish connection to the client: %s", strerror (errno));
        goto error;
    }

    g_debug ("Client connection setup successfull");
    return 0;
error:
    rdma_reject (client, NULL, 0);
    kiro_destroy_connection_context (&ctx);
    rdma_destroy_id (client);
    return -1;
}


static int
welcome_client (struct rdma_cm_id *client, void *mem, size_t mem_size)
{
    struct kiro_connection_context *ctx = (struct kiro_connection_context *) (client->context);
    ctx->rdma_mr = (struct kiro_rdma_mem *)g_try_malloc0 (sizeof (struct kiro_rdma_mem));

    if (!ctx->rdma_mr) {
        g_critical ("Failed to allocate RDMA Memory Container: %s", strerror (errno));
        return -1;
    }

    ctx->rdma_mr->mem = mem;
    ctx->rdma_mr->size = mem_size;
    ctx->rdma_mr->mr = rdma_reg_read (client, ctx->rdma_mr->mem, ctx->rdma_mr->size);

    if (!ctx->rdma_mr->mr) {
        g_critical ("Failed to register RDMA Memory Region: %s", strerror (errno));
        kiro_destroy_rdma_memory (ctx->rdma_mr);
        return -1;
    }

    struct kiro_ctrl_msg *msg = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);

    msg->msg_type = KIRO_ACK_RDMA;

    msg->peer_mri = * (ctx->rdma_mr->mr);

    if (rdma_post_send (client, client, ctx->cf_mr_send->mem, ctx->cf_mr_send->size, ctx->cf_mr_send->mr, IBV_SEND_SIGNALED)) {
        g_warning ("Failure while trying to post SEND: %s", strerror (errno));
        kiro_destroy_rdma_memory (ctx->rdma_mr);
        return -1;
    }

    struct ibv_wc wc;

    if (rdma_get_send_comp (client, &wc) < 0) {
        g_warning ("Failed to post RDMA MRI to client: %s", strerror (errno));
        kiro_destroy_rdma_memory (ctx->rdma_mr);
        return -1;
    }

    g_debug ("RDMA MRI sent to client");
    return 0;
}


static gboolean
process_rdma_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    //(void) condition;
    g_debug ("Message condition: %i", condition);

    struct kiro_client_connection *cc = (struct kiro_client_connection *)data;
    struct ibv_wc wc;

    if (ibv_poll_cq (cc->conn->recv_cq, 1, &wc) < 0) {
        g_critical ("Failure getting receive completion event from the queue: %s", strerror (errno));
        return FALSE;
    }
    void *cq_ctx;
    struct ibv_cq *cq;
    int err = ibv_get_cq_event (cc->conn->recv_cq_channel, &cq, &cq_ctx);
    if (!err)
        ibv_ack_cq_events (cq, 1);

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)cc->conn->context;
    guint type = ((struct kiro_ctrl_msg *)ctx->cf_mr_recv->mem)->msg_type;
    g_debug ("Received a message from Client %u of type %u", cc->id, type);

    if (type == KIRO_PING) {
        struct kiro_ctrl_msg *msg = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
        msg->msg_type = KIRO_PONG;

        if (rdma_post_send (cc->conn, cc->conn, ctx->cf_mr_send->mem, ctx->cf_mr_send->size, ctx->cf_mr_send->mr, IBV_SEND_SIGNALED)) {
            g_warning ("Failure while trying to post PONG send: %s", strerror (errno));
            goto done;
        }

        if (rdma_get_send_comp (cc->conn, &wc) < 0) {
            g_warning ("An error occured while sending PONG: %s", strerror (errno));
        }
    }

done:
    //Post a generic receive in order to stay responsive to any messages from
    //the client
    if (rdma_post_recv (cc->conn, cc->conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        //TODO: Connection teardown in an event handler routine? Not a good
        //idea...
        g_critical ("Posting generic receive for event handling failed: %s", strerror (errno));
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (cc->conn);
        return FALSE;
    }

    ibv_req_notify_cq (cc->conn->recv_cq, 0); // Make the respective Queue push events onto the channel

    g_debug ("Finished RDMA event handling");
    return TRUE;
}


static gboolean
process_cm_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    (void) condition;

    KiroServerPrivate *priv = (KiroServerPrivate *)data;
    struct rdma_cm_event *active_event;

    if (0 <= rdma_get_cm_event (priv->ec, &active_event)) {
        //Disable cancellation to prevent undefined states during shutdown
        struct rdma_cm_event *ev = g_try_malloc (sizeof (*active_event));

        if (!ev) {
            g_critical ("Unable to allocate memory for Event handling!");
            rdma_ack_cm_event (active_event);
            return FALSE;
        }

        memcpy (ev, active_event, sizeof (*active_event));
        rdma_ack_cm_event (active_event);

        if (ev->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
            if (TRUE == priv->close_signal) {
                //Main thread has signalled shutdown!
                //Don't connect this client any more.
                //Sorry mate!
                rdma_reject (ev->id, NULL, 0);
                return TRUE;
            }

            do {
                g_debug ("Got connection request from client");
                struct kiro_client_connection *cc = (struct kiro_client_connection *)g_try_malloc (sizeof (struct kiro_client_connection));
                if (!cc) {
                    errno = ENOMEM;
                    rdma_reject (ev->id, NULL, 0);
                    goto fail;
                }

                if (connect_client (ev->id))
                    goto fail;

                // Post a welcoming "Receive" for handshaking
                if (welcome_client (ev->id, priv->mem, priv->mem_size))
                    goto fail;

                ibv_req_notify_cq (ev->id->recv_cq, 0); // Make the respective Queue push events onto the channel

                // Connection set-up successfully! (Server)
                // ctx was created by 'welcome_client'
                struct kiro_connection_context *ctx = (struct kiro_connection_context *) (ev->id->context);
                ctx->identifier = priv->next_client_id++;
                ctx->container = cc; // Make the connection aware of its container

                // Fill the client connection container. Also create a
                // g_io_channel wrapper for the new clients receive queue event
                // channel and add a main_loop watch to it.
                cc->id = ctx->identifier;
                cc->conn = ev->id;
                cc->rcv_ec = g_io_channel_unix_new (ev->id->recv_cq_channel->fd);
                cc->source_id = g_io_add_watch (cc->rcv_ec, G_IO_IN | G_IO_PRI, process_rdma_event, (gpointer)cc);
                g_io_channel_unref (cc->rcv_ec); // main_loop now holds a reference. We don't need ours any more

                priv->clients = g_list_append (priv->clients, (gpointer)cc);
                g_debug ("Client connection assigned with ID %u", ctx->identifier);
                g_debug ("Currently %u clients in total are connected", g_list_length (priv->clients));
                break;

                fail:
                    g_warning ("Failed to accept client connection: %s", strerror (errno));

            } while(0);
        }
        else if (ev->event == RDMA_CM_EVENT_DISCONNECTED) {
            struct kiro_connection_context *ctx = (struct kiro_connection_context *) (ev->id->context);
            if (!ctx->container) {
                g_debug ("Got disconnect request from unknown client");
                return FALSE;
            }

            GList *client = g_list_find (priv->clients, (gconstpointer) ctx->container);

            if (client) {
                g_debug ("Got disconnect request from client ID %u", ctx->identifier);
                struct kiro_client_connection *cc = (struct kiro_client_connection *)ctx->container;
                g_source_remove (cc->source_id); // this also unrefs the GIOChannel of the source. Nice.
                priv->clients = g_list_delete_link (priv->clients, client);
                g_free (cc);
            }
            else
                g_debug ("Got disconnect request from unknown client");

            // Note:
            // The ProtectionDomain needs to be buffered and freed manually.
            // Each connecting client is attached with its own pd, which we
            // create manually. So we also need to clean it up manually.
            // This needs to be done AFTER the connection is brought down, so we
            // buffer the pointer to the pd and clean it up afterwards.
            struct ibv_pd *pd = ev->id->pd;
            kiro_destroy_connection (& (ev->id));
            g_free (pd);

            g_debug ("Connection closed successfully. %u connected clients remaining", g_list_length (priv->clients));
        }

        g_free (ev);
    }
    return TRUE;
}


gpointer
start_server_main_loop (gpointer data)
{
    g_main_loop_run ((GMainLoop *)data);
    return NULL;
}


int
kiro_server_start (KiroServer *self, const char *address, const char *port, void *mem, size_t mem_size)
{
    KiroServerPrivate *priv = KIRO_SERVER_GET_PRIVATE (self);

    if (priv->base) {
        g_debug ("Server already started.");
        return -1;
    }

    if (!mem || mem_size == 0) {
        g_warning ("Invalid memory given to provide.");
        return -1;
    }

    struct rdma_addrinfo hints, *res_addrinfo;
    memset (&hints, 0, sizeof (hints));
    hints.ai_port_space = RDMA_PS_IB;
    hints.ai_flags = RAI_PASSIVE;

    char *addr_c = g_strdup (address);
    char *port_c = g_strdup (port);

    int rtn = rdma_getaddrinfo (addr_c, port_c, &hints, &res_addrinfo);
    g_free (addr_c);
    g_free (port_c);

    if (rtn) {
        g_critical ("Failed to create address information: %s", strerror (errno));
        return -1;
    }

    struct ibv_qp_init_attr qp_attr;
    memset (&qp_attr, 0, sizeof (qp_attr));
    qp_attr.cap.max_send_wr = 10;
    qp_attr.cap.max_recv_wr = 10;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.qp_context = priv->base;
    qp_attr.sq_sig_all = 1;

    if (rdma_create_ep (& (priv->base), res_addrinfo, NULL, &qp_attr)) {
        g_critical ("Endpoint creation failed: %s", strerror (errno));
        g_free (res_addrinfo);
        return -1;
    }
    g_free (res_addrinfo); // No longer needed

    g_debug ("Endpoint created");
    char *addr_local = NULL;
    struct sockaddr *src_addr = rdma_get_local_addr (priv->base);

    if (!src_addr) {
        addr_local = "NONE";
    }
    else {
        addr_local = inet_ntoa (((struct sockaddr_in *)src_addr)->sin_addr);
        /*
        if(src_addr->sa_family == AF_INET)
            addr_local = &(((struct sockaddr_in*)src_addr)->sin_addr);
        else
            addr_local = &(((struct sockaddr_in6*)src_addr)->sin6_addr);
        */
    }

    g_message ("Server bound to address %s:%s", addr_local, port);

    if (rdma_listen (priv->base, 0)) {
        g_critical ("Failed to put server into listening state: %s", strerror (errno));
        rdma_destroy_ep (priv->base);
        return -1;
    }

    priv->mem = mem;
    priv->mem_size = mem_size;
    priv->ec = rdma_create_event_channel();

    if (rdma_migrate_id (priv->base, priv->ec)) {
        g_critical ("Was unable to migrate connection to new Event Channel: %s", strerror (errno));
        rdma_destroy_ep (priv->base);
        return -1;
    }

    priv->main_loop = g_main_loop_new (NULL, FALSE);
    priv->conn_ec = g_io_channel_unix_new (priv->ec->fd);
    g_io_add_watch (priv->conn_ec, G_IO_IN | G_IO_PRI, process_cm_event, (gpointer)priv);
    priv->main_thread = g_thread_new ("KIRO Server main loop", start_server_main_loop, priv->main_loop);

    // We gave control to the main_loop (with add_watch) and don't need our ref
    // any longer
    g_io_channel_unref (priv->conn_ec);


    g_message ("Enpoint listening");
    return 0;
}


static void
disconnect_client (gpointer data, gpointer user_data)
{
    (void)user_data;

    if (data) {
        struct kiro_client_connection *cc = (struct kiro_client_connection *)data;
        struct rdma_cm_id *id = cc->conn;
        struct kiro_connection_context *ctx = (struct kiro_connection_context *) (id->context);
        g_debug ("Disconnecting client: %u", ctx->identifier);
        g_source_remove (cc->source_id);

        // Note:
        // The ProtectionDomain needs to be buffered and freed manually.
        // Each connecting client is attached with its own pd, which we
        // create manually. So we also need to clean it up manually.
        // This needs to be done AFTER the connection is brought down, so we
        // buffer the pointer to the pd and clean it up afterwards.
        struct ibv_pd *pd = cc->conn->pd;
        kiro_destroy_connection (&(cc->conn));
        g_free (cc);
        g_free (pd);
    }
}


void
kiro_server_stop (KiroServer *self)
{
    if (!self)
        return;

    KiroServerPrivate *priv = KIRO_SERVER_GET_PRIVATE (self);

    if (!priv->base)
        return;

    //Shut down event listening
    priv->close_signal = TRUE;
    g_debug ("Event handling stopped");

    g_list_foreach (priv->clients, disconnect_client, NULL);
    g_list_free (priv->clients);

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
    g_io_channel_unref (priv->conn_ec);
    priv->conn_ec = NULL;
    priv->close_signal = FALSE;

    // kiro_destroy_connection would try to call rdma_disconnect on the given
    // connection. But the server never 'connects' to anywhere, so this would
    // cause a crash. We need to destroy the enpoint manually without disconnect
    struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->base->context);
    kiro_destroy_connection_context (&ctx);
    rdma_destroy_ep (priv->base);
    priv->base = NULL;

    rdma_destroy_event_channel (priv->ec);
    priv->ec = NULL;
    g_message ("Server stopped successfully");
}

