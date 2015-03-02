/* Copyright (C) 2014-2015 Timo Dritschler <timo.dritschler@kit.edu>
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
 * SECTION: kiro-sb
 * @Short_description: KIRO 'Synchronizing Buffer'
 * @Title: KiroSb
 *
 * KiroSb implements a 'Synchronizing Buffer' that automatically keeps the local
 * memory content up to date by mirroring the remote SyncBuffers memory content
 * automatically without any required user interaction
 */

#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include "kiro-sb.h"
#include "kiro-trb.h"
#include "kiro-server.h"
#include "kiro-client.h"


/*
 * Definition of 'private' structures and members and macro to access them
 */

#define KIRO_SB_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), KIRO_TYPE_SB, KiroSbPrivate))

struct _KiroSbPrivate {

    /* Properties */
    // PLACEHOLDER //

    /* 'Real' private structures */
    /* (Not accessible by properties) */
    int         initialized;    // 0 if uninitialized, 1 if server, 2 if client
    KiroServer* server;         // KIRO Server component to serve
    KiroClient* client;         // KIRO Client component to clone
    KiroTrb* trb;               // KIRO Ring Buffer to hold and exchange data

    GThread     *main_thread;   // Main thread for the main_loop
    GMainLoop   *main_loop;     // main_loop *duh*
    guint       close_signal;   // Used to signal shutdown of the main_loop
    gboolean    freeze;         // Allows to prevent auto-sync

    GHookList   callbacks;      // List of registerd sync-callbacks
};


G_DEFINE_TYPE (KiroSb, kiro_sb, G_TYPE_OBJECT);


KiroSb *
kiro_sb_new (void)
{
    return g_object_new (KIRO_TYPE_SB, NULL);
}


void
kiro_sb_free (KiroSb *sb)
{
    g_return_if_fail (sb != NULL);
    if (KIRO_IS_SB (sb))
        g_object_unref (sb);
    else
        g_warning ("Trying to use kiro_sb_free on an object which is not a KIRO SB. Ignoring...");
}


static void
kiro_sb_init (KiroSb *self)
{
    g_return_if_fail (self != NULL);
    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);
    priv->initialized = 0;
    priv->trb = NULL;
    priv->server = NULL;
    priv->client = NULL;
    priv->freeze = FALSE;
    g_hook_list_init (&(priv->callbacks), sizeof (GHook));
}


static void
kiro_sb_finalize (GObject *object)
{
    g_return_if_fail (object != NULL);
    KiroSb *self = KIRO_SB (object);

    kiro_sb_stop (self);

    G_OBJECT_CLASS (kiro_sb_parent_class)->finalize (object);
}


static void
kiro_sb_class_init (KiroSbClass *klass)
{
    g_return_if_fail (klass != NULL);
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    gobject_class->finalize = kiro_sb_finalize;
    g_type_class_add_private (klass, sizeof (KiroSbPrivate));
}


void
kiro_sb_stop (KiroSb *self)
{
    g_return_if_fail (self != NULL);
    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);

    g_return_if_fail (priv->initialized != 0);

    if (priv->initialized == 1) {
        if (priv->server)
            kiro_server_free (priv->server);
    }

    if (priv->initialized == 2) {
        priv->close_signal = TRUE;
        while (g_main_loop_is_running (priv->main_loop)) {}
        g_thread_join (priv->main_thread);
        g_thread_unref (priv->main_thread);
        priv->main_thread = NULL;

        if (priv->client)
            kiro_client_free (priv->client);
    }

    g_hook_list_clear (&(priv->callbacks));

    if (priv->trb) {
        kiro_trb_purge (priv->trb, FALSE);
        kiro_trb_free (priv->trb);
    }

    priv->trb = NULL;
    priv->server = NULL;
    priv->client = NULL;
    priv->initialized = 0;
}


gpointer
start_main_loop (GMainLoop *loop)
{
    g_main_loop_run (loop);
    /* wait for mai loop to finish*/
    g_main_loop_unref (loop);
    return NULL;
}


gboolean
idle_func (KiroSbPrivate *priv)
{
    if (priv->close_signal) {
        g_main_loop_quit (priv->main_loop);
        /*main_thread will do the unref upon exit*/
        priv->main_loop = NULL;
        g_debug ("Main loop quit");
        return G_SOURCE_REMOVE;
    }

    if (TRUE == priv->freeze)
        return G_SOURCE_CONTINUE;

    struct KiroTrbInfo *header = (struct KiroTrbInfo *)kiro_trb_get_raw_buffer (priv->trb);
    gulong old_offset = header->offset;
    kiro_client_sync_partial (priv->client, 0, sizeof(struct KiroTrbInfo), 0);
    kiro_trb_refresh (priv->trb);
    if ((old_offset != header->offset) && 0 < header->offset) {
        gulong offset = (gulong) (kiro_trb_get_element (priv->trb, -1) - kiro_trb_get_raw_buffer (priv->trb));
        kiro_client_sync_partial (priv->client, offset, kiro_trb_get_element_size (priv->trb), offset);
        g_hook_list_invoke_check (&(priv->callbacks), FALSE);
    }

    return G_SOURCE_CONTINUE;
}


void
kiro_sb_freeze (KiroSb *self)
{
    g_return_if_fail (self != NULL);
    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);

    priv->freeze = TRUE;
}


void
kiro_sb_thaw (KiroSb *self)
{
    g_return_if_fail (self != NULL);
    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);

    priv->freeze = FALSE;
}


gboolean
kiro_sb_serve (KiroSb *self, gulong size)
{
    g_return_val_if_fail (self != NULL, FALSE);

    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);
    g_return_val_if_fail (priv->initialized == 0, FALSE);

    g_return_val_if_fail ((priv->trb = kiro_trb_new ()), FALSE);

    if (0 > kiro_trb_reshape (priv->trb, size, 3)) {
        g_debug ("Failed to create KIRO ring buffer");
        kiro_trb_free (priv->trb);
        return FALSE;
    }

    void *buff = kiro_trb_get_raw_buffer (priv->trb);
    gulong b_size = kiro_trb_get_raw_size (priv->trb);

    priv->server = kiro_server_new ();
    if (0 > kiro_server_start (priv->server, NULL, "60010", buff, b_size)) {
        g_debug ("Failed to start KIRO Server");
        kiro_server_free (priv->server);
        kiro_trb_free (priv->trb);
        return FALSE;
    }

    priv->initialized = 1;
    g_message ("SyncBuffer ready");

    return TRUE;
}


KiroContinueFlag
ready_callback (gboolean *ready)
{
    *ready = TRUE;
    return KIRO_CALLBACK_REMOVE;
}


void *
kiro_sb_get_data_blocking (KiroSb *self)
{
    g_return_val_if_fail (self != NULL, NULL);

    gboolean *ready = g_malloc(sizeof (gboolean));
    if (!ready)
        return NULL;
    *ready = FALSE;
    kiro_sb_add_sync_callback (self, (KiroSbSyncCallbackFunc)ready_callback, ready);

    while (!(*ready)) {}
    g_free (ready);
    return kiro_sb_get_data (self);
}


void *
kiro_sb_get_data (KiroSb *self)
{
    g_return_val_if_fail (self != NULL, NULL);
    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);

    struct KiroTrbInfo *header = kiro_trb_get_raw_buffer (priv->trb);
    switch (header->offset) {
        case 0:
            return kiro_trb_get_element (priv->trb, 0);
            break;
        case 1:
            return kiro_trb_get_element (priv->trb, 1);
            break;
        default:
            return kiro_trb_get_element (priv->trb, -1);
    }
}


gboolean
kiro_sb_push (KiroSb *self, void *data_in)
{
    g_return_val_if_fail (self != NULL, FALSE);

    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);
    g_return_val_if_fail (priv->initialized != 1, FALSE);

    return kiro_trb_push (priv->trb, data_in);
}


void *
kiro_sb_push_dma (KiroSb *self)
{
    g_return_val_if_fail (self != NULL, FALSE);

    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);
    g_return_val_if_fail (priv->initialized != 1, FALSE);

    return kiro_trb_dma_push (priv->trb);
}


gboolean
kiro_sb_clone (KiroSb *self, const gchar* address, const gchar* port)
{
    g_return_val_if_fail (self != NULL, FALSE);

    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);
    g_return_val_if_fail (priv->initialized == 0, FALSE);

    g_return_val_if_fail ((priv->trb = kiro_trb_new ()), FALSE);

    priv->client = kiro_client_new ();
    if (0 > kiro_client_connect (priv->client, address, port)) {
        g_debug ("Failed to connect to remote Sync Buffer");
        kiro_trb_free (priv->trb);
        kiro_client_free (priv->client);
        return FALSE;
    }

    kiro_client_sync (priv->client);
    kiro_trb_adopt (priv->trb, kiro_client_get_memory (priv->client));

    priv->main_loop = g_main_loop_new (NULL, FALSE);
    g_idle_add ((GSourceFunc)idle_func, priv);
    priv->main_thread = g_thread_new ("KIRO SB Main Loop", (GThreadFunc)start_main_loop, priv->main_loop);

    priv->initialized = 2;
    return TRUE;
}


gulong
kiro_sb_add_sync_callback (KiroSb *self, KiroSbSyncCallbackFunc func, void *user_data)
{
    g_return_val_if_fail (self != NULL, 0);

    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);

    GHook *new_hook = g_hook_alloc (&(priv->callbacks));
    new_hook->data = user_data;
    new_hook->func = (GHookCheckFunc)func;
    g_hook_append (&(priv->callbacks), new_hook);
    return new_hook->hook_id;
}


gboolean
kiro_sb_remove_sync_callback (KiroSb *self, gulong hook_id)
{
    g_return_val_if_fail (self != NULL, FALSE);
    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);

    return g_hook_destroy (&(priv->callbacks), hook_id);
}


void
kiro_sb_clear_sync_callbacks (KiroSb *self)
{
    g_return_val_if_fail (self != NULL, FALSE);
    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);

    g_hook_list_clear (&(priv->callbacks));
}


gulong
kiro_sb_get_size (KiroSb *self)
{
    g_return_val_if_fail (self != NULL, 0);

    KiroSbPrivate *priv = KIRO_SB_GET_PRIVATE (self);
    g_return_val_if_fail (priv->initialized != 0, 0);

    return kiro_trb_get_element_size (priv->trb);
}
