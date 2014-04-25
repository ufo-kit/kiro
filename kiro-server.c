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
    struct rdma_event_channel   *ec;        // Main Event Channel
    struct rdma_cm_id           *base;      // Base-Listening-Connection
    struct kiro_connection      *client;    // Connection to the client
    KiroTrb                     *buffer;    // Memory Container
    

};


G_DEFINE_TYPE_WITH_PRIVATE (KiroServer, kiro_server, G_TYPE_OBJECT);


static void kiro_server_init (KiroServer *self)
{
    KiroServerPrivate *priv = KIRO_SERVER_GET_PRIVATE(self);
    memset(priv, 0, sizeof(&priv));
    
    priv->buffer = g_object_new(KIRO_TYPE_TRB, NULL);
    kiro_trb_reshape(priv->buffer, sizeof(uint64_t), 1000);
    uint64_t a = 0xAFFED00F;
    uint64_t b = 0x1337BEEF;
    kiro_trb_push(priv->buffer, &a);
    kiro_trb_push(priv->buffer, &b);
}

static void
kiro_server_finalize (GObject *object)
{
    KiroServer *self = KIRO_SERVER(object);
    KiroServerPrivate * priv = KIRO_SERVER_GET_PRIVATE(self);
    g_object_unref(priv->buffer);
}

static void
kiro_server_class_init (KiroServerClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    gobject_class->finalize = kiro_server_finalize;
}



int kiro_server_start (KiroServer *self, char *address, char *port)
{
    KiroServerPrivate *priv = KIRO_SERVER_GET_PRIVATE(self);

    if(priv->base)
    {
        printf("Server already started.\n");
        return -1;
    }
    
    struct rdma_addrinfo hints, *res_addrinfo;
    memset(&hints, 0, sizeof(hints));
    hints.ai_port_space = RDMA_PS_IB;
    hints.ai_flags = RAI_PASSIVE;
    if(rdma_getaddrinfo(address, port, &hints, &res_addrinfo))
    {
        printf("Failed to bind to address %s:%s\n",address, port);
        return -1;
    }
    printf("Bound to address %s:%s\n",address, port);
    
    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.cap.max_send_wr = 10;
    qp_attr.cap.max_recv_wr = 10;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.qp_context = priv->base;
    qp_attr.sq_sig_all = 1;
    
    if(rdma_create_ep(&(priv->base), res_addrinfo, NULL, &qp_attr))
    {
        printf("Endpoint creation failed.\n");
        return -1;
    }
    printf("Endpoint created.\n");
    
    if(rdma_listen(priv->base, 0))
    {
        printf("Failed to put server into listening state.\n");
        rdma_destroy_ep(priv->base);
        return -1;
    }
    printf("Enpoint listening.\n");
    
    priv->client = (struct kiro_connection *)calloc(1, sizeof(struct kiro_connection));
    if(!priv->client)
    {
        printf("Failed to create container for client connection.\n");
        rdma_destroy_ep(priv->base);
        return -1;
    }
    priv->client->identifier = 0; //First Client
    
    printf("Waiting for connection request.\n");
    if(rdma_get_request(priv->base, &(priv->client->id)))
    {
        printf("Failure waiting for clienet connection.\n");
        rdma_destroy_ep(priv->base);
        return -1;
    }
    printf("Connection Request received.\n");
    
    struct kiro_connection_context *ctx = (struct kiro_connection_context *)calloc(1,sizeof(struct kiro_connection_context));
    if(!ctx)
    {
        printf("Failed to create connection context.\n");
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    ctx->cf_mr_send = (struct kiro_rdma_mem *)calloc(1, sizeof(struct kiro_rdma_mem));
    ctx->cf_mr_recv = (struct kiro_rdma_mem *)calloc(1, sizeof(struct kiro_rdma_mem));
    if(!ctx->cf_mr_recv || !ctx->cf_mr_send)
    {
        printf("Failed to allocate Control Flow Memory Container.\n");
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    ctx->cf_mr_recv = kiro_create_rdma_memory(priv->client->id->pd, sizeof(struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);
    ctx->cf_mr_send = kiro_create_rdma_memory(priv->client->id->pd, sizeof(struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);
    if(!ctx->cf_mr_recv || !ctx->cf_mr_send)
    {
        printf("Failed to register control message memory.\n");
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    ctx->cf_mr_recv->size = ctx->cf_mr_send->size = sizeof(struct kiro_ctrl_msg);
    priv->client->id->context = ctx;
    
    if(rdma_post_recv(priv->client->id, priv->client, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr))
    {
        printf("Posting preemtive receive for connection failed.\n");
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    if(rdma_accept(priv->client->id, NULL))
    {
        printf("Failed to establish connection to the server.\n");
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    priv->ec = rdma_create_event_channel();
    int oldflags = fcntl (priv->ec->fd, F_GETFL, 0);
    /* Only change the FD Mode if we were able to get its flags */
    if (oldflags >= 0) {
        oldflags |= O_NONBLOCK;
        /* Store modified flag word in the descriptor. */
        fcntl (priv->ec->fd, F_SETFL, oldflags);
    }
    if(rdma_migrate_id(priv->base, priv->ec))
    {
        printf("Was unable to migrate connection to new Event Channel.\n");
        rdma_disconnect(priv->client->id);
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    printf("Client Connected.\n");
    
        
    ctx->rdma_mr = (struct kiro_rdma_mem *)calloc(1, sizeof(struct kiro_rdma_mem));
    if(!ctx->rdma_mr)
    {
        printf("Failed to allocate RDMA Memory Container.\n");
        rdma_disconnect(priv->client->id);
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    ctx->rdma_mr->mem = kiro_trb_get_raw_buffer(priv->buffer);
    ctx->rdma_mr->size = kiro_trb_get_raw_size(priv->buffer);
    ctx->rdma_mr->mr = rdma_reg_read(priv->client->id, ctx->rdma_mr->mem, ctx->rdma_mr->size);
    if(!ctx->rdma_mr->mr)
    {
        printf("Failed to register RDMA Memory Region.\n");
        rdma_disconnect(priv->client->id);
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    struct kiro_ctrl_msg *msg = (struct kiro_ctrl_msg *)(ctx->cf_mr_send->mem);
    msg->msg_type = KIRO_ACK_RDMA;
    msg->peer_mri = *(ctx->rdma_mr->mr);
    
    if(rdma_post_send(priv->client->id, priv->client, ctx->cf_mr_send->mem, ctx->cf_mr_send->size, ctx->cf_mr_send->mr, IBV_SEND_SIGNALED))
    {
        printf("Failure while trying to post SEND.\n");
        rdma_disconnect(priv->client->id);
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    
    struct ibv_wc wc;
    
    if(rdma_get_send_comp(priv->client->id, &wc) < 0)
    {
        printf("Failed to post RDMA MRI to client.\n");
        rdma_disconnect(priv->client->id);
        kiro_destroy_connection_context(ctx);
        rdma_destroy_ep(priv->base);
        rdma_destroy_ep(priv->client->id);
        free(priv->client);
        return -1;
    }
    printf("RDMA MRI sent to client.\n");
    sleep(1);
    return 0;
}



int kiro_server_sync (KiroServer *self)
{
    //PASS
    return 0;
}










