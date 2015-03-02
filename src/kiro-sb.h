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
 * SECTION: kiro-sync-buffer
 * @Short_description: KIRO 'Synchronizing Buffer'
 * @Title: KiroSb
 *
 * KiroSb implements a 'Synchronizing Buffer' that automatically keeps the local
 * memory content up to date by mirroring the remote SyncBuffers memory content
 * automatically without any required user interaction
 */

#ifndef __KIRO_SB_H
#define __KIRO_SB_H

#include <stdint.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define KIRO_TYPE_SB             (kiro_sb_get_type())
#define KIRO_SB(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), KIRO_TYPE_SB, KiroSb))
#define KIRO_IS_SB(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), KIRO_TYPE_SB))
#define KIRO_SB_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), KIRO_TYPE_SB, KiroSbClass))
#define KIRO_IS_SB_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), KIRO_TYPE_SB))
#define KIRO_SB_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj), KIRO_TYPE_SB, KiroSbClass))


typedef struct _KiroSb           KiroSb;
typedef struct _KiroSbClass      KiroSbClass;
typedef struct _KiroSbPrivate    KiroSbPrivate;


struct _KiroSb {

    GObject parent;

};


/**
 * IbvConnectorInterface:
 *
 * Base interface for IbvConnectors.
 */

struct _KiroSbClass {

    GObjectClass parent_class;

};


/* GObject and GType functions */
/**
 * kiro_sb_get_type: (skip)
 * Returns: GType of #KiroSb
 */
GType       kiro_sb_get_type           (void);

/**
 * kiro_sb_new - Creates a new #KiroSb
 * Returns: (transfer full): A pointer to a new #KiroSb
 * Description:
 *   Creates a new #KiroSb and returns a pointer to it.
 * See also:
 *   kiro_sb_free
 */
KiroSb*    kiro_sb_new                (void);

/**
 * kiro_sb_free - 'Destroys' the given #KiroSb
 * @trb: (transfer none): The #KiroSb that is to be freed
 * Description:
 *   Clears all underlying memory and frees the object memory. 
 * Note:
 *   The internal memory is also freed when calling this function. If you want
 *   to continue using the raw @sb memory after you call this function, you need
 *   to memcpy() its content using the information optained from
 *   kiro_sb_get_element()
 * See also:
 *   kiro_sb_new, kiro_sb_get_element
 */
void        kiro_sb_free               (KiroSb *sb);

/**
 * kiro_sb_stop - Stops the #KiroSb and clears all internal memory
 * @sb: (transfer none): The #KiroSb to stop
 * Description:
 *   The given #KiroSb is stopped and all internal memory is cleared. It is put
 *   back into its initial state and it can be used as if it was just created
 * See also:
 *   kiro_sb_new, kiro_sb_serve, kiro_sb_clone
 */
void    kiro_sb_stop    (KiroSb *sb);


typedef gboolean KiroContinueFlag;
#define KIRO_CALLBACK_CONTINUE TRUE
#define KIRO_CALLBACK_REMOVE FALSE

/**
 * KiroSbSyncCallbackFunc - Function type for sync callbacks os a #KiroSb
 * @user_data: (transfer none): The #user_data which was provided during
 * registration of this callback
 * Returns: A #KiroContinueFlag
 * Description:
 *   Defines the type of a callback function which will be invoked every time
 *   the #KiroSb syncs new data
 * Note:
 *   Returning %FALSE or %KIRO_CALLBACK_REMOVE will automatically remove the callback
 *   from the internal callback list. Return %TRUE or %KIRO_CALLBACK_CONTINUE if you
 *   want to keep the callback active
 * See also:
 *   kiro_sb_add_sync_callback, kiro_sb_remove_sync_callback, kiro_sb_clear_sync_callbacks
 */
typedef KiroContinueFlag (*KiroSbSyncCallbackFunc)   (void *user_data);

/**
 * kiro_sb_add_sync_callback - Add a sync callback to the #KiroSb
 * @sb: (tranfer none): The #KiroSb to register this callback to
 * @func: (transfer none): A function pointer to the callback function
 * Returns: The internal id of the registerd callback
 * Description:
 *   Adds a callback to the passed #KiroSbSyncCallbackFunc to this #KiroSb which
 *   will be invoked every time the #KiroSb syncs new data.
 * Note:
 *   The sync callbacks will only be invoked on a 'clonig' #KiroSb. All
 *   registered callbacks will be invoked in the order they were added to the
 *   #KiroSb.
 * See also:
 *   kiro_sb_remove_sync_callback, kiro_sb_clear_sync_callbacks
 */
gulong    kiro_sb_add_sync_callback (KiroSb *sb, KiroSbSyncCallbackFunc callback, void *user_data);

/**
 * kiro_sb_remove_sync_callback - Remove a callback from the list
 * @sb: (transfer none): The #KiroSb to remove the callback from
 * @id: The id of the callback to be removed
 * Returns: A #gboolean. %TRUE if the callback was found and removed. %FALSE
 *   otherwise
 * Description:
 *   Removes the callback with the given @id from the internal list. If the
 *   callback with the given @id was not found %FALSE is returned. If the
 *   callback with the given @id was found, it will be removed from the callback
 *   list and %TRUE is returned
 * Note:
 *   Any currently active callback will still finish before it is removed from
 *   the list.
 * See also:
 *   kiro_sb_add_sync_callback, kiro_sb_clear_sync_callbacks
 */
gboolean    kiro_sb_remove_sync_callback (KiroSb *sb, gulong id);

/**
 * kiro_sb_clear_sync_callbacks - Clear all sync callbacks
 * @sb: (transfer none): The #KiroSb to perform this operation on
 * Description:
 *   Removes all registerd callbacks from the internal list
 * Note:
 *   Any currently active callbacks will still finish before they are removed
 *   from the list
 * See also:
 *   kiro_sb_add_sync_callback, kiro_sb_remove_sync_callback
 */
void    kiro_sb_clear_sync_callbacks (KiroSb *sb);

/**
 * kiro_sb_serve - Allow remote KiroSbs to clone this buffers memory
 * Returns: A gboolean. TRUE = success. FALSE = fail.
 * @sb: (transfer none): The #KiroSb to perform this operation on
 * @size: Size in bytes of the content that will be served
 * Description:
 *   Allows other remote #KiroSbs to connect to this #KiroSb and clone its
 *   memory. The internal memory is initially empty. Use the kiro_sb_push or
 *   kiro_sb_push_dma functions to update the served data.
 * Note:
 *   A #KiroSb that already 'serves' its content can no longer clone
 *   other remote #KiroSbs.
 * See also:
 *   kiro_sb_push, kiro_sb_push_dma
 */
gboolean    kiro_sb_serve           (KiroSb *sb, gulong size);

/**
 * kiro_sb_clone - Clone the content of a remote #KiroSb
 * Returns: A gboolean. TRUE = connection successful. FALSE = connection failed.
 * @sb: (transfer none): The #KiroSb to perform this operation on
 * @address: The InfiniBand address of the remote #KiroSb which should be cloned
 * @port: The InfiniBand port of the remote #KiroSb which should be cloned
 * Description:
 *   Connects to the remote #KiroSb given by @address and @port and
 *   continuousely clones its content into the local #KiroSb
 * Note:
 *   A #KiroSb that clones a remote #KiroSb can no longer start to 'serve' its
 *   content to other remote #KiroSbs
 * See also:
 *
 */
gboolean    kiro_sb_clone       (KiroSb *sb, const gchar *address, const gchar *port);

/**
 * kiro_sb_get_size - Get the size in bytes of the managed memory
 * Returns: A gulong giving the size of the managed memory in bytes
 * @sb: (transfer none): The #KiroSb to perform this operation on
 * Description:
 *   Returns the size in bytes of the content that is being served and/or cloned
 *   from.
 * Note:
 *   Since #KiroSb uses an internal triple buffer, the value gained from this
 *   function only gives the size of one element from that buffer. The size of
 *   the entire data structure will be different.
 * See also:
 *
 */
gulong      kiro_sb_get_size    (KiroSb *sb);

/**
 * kiro_sb_freeze - Stop auto syncing
 * @sb: (transfer none): The #KiroSb to perform this operation on
 * Description:
 *   Stops the given #KiroSb from automatically syncing.
 * See also:
 *   kiro_sb_thaw
 */
void      kiro_sb_freeze      (KiroSb *sb);

/**
 * kiro_sb_thaw - Enable auto syncing
 * @sb: (transfer none): The #KiroSb to perform this operation on
 * Description:
 *   Enable the given #KiroSb automatic syncing.
 * See also:
 *   kiro_sb_freeze
 */
void     kiro_sb_thaw         (KiroSb *sb);

/**
 * kiro_sb_get_data - Get a pointer to the stored data
 * @sb: (transfer none) The #KiroSb to get the data from
 * Returns: A void pointer the stored data
 * Description:
 *   Returns a void pointer to the most current incarnation of the stored data.
 *   Data might either change by pushing (in case of a 'serving' #KiroSb) or
 *   after (automatic or manual) syncing (in case of a 'cloning' #KiroSb).
 * Note:
 *   The returned pointer to the element might become invalid at any time by
 *   automatic or manual sync. Under no circumstances might the returned pointer
 *   be freed by the user. If you want to ensure access to the pointed-to data
 *   after a sync, you should use memcpy().
 * See also:
 *   kiro_sb_freeze, kiro_sb_serve, kiro_sb_clone, kiro_sb_push,
 *   kiro_sb_push_dma, kiro_sb_get_data_blocking
 */
void*   kiro_sb_get_data     (KiroSb *sb);

/**
 * kiro_sb_get_data_blocking - Wait for new data and return a pointer to it
 * @sb: (transfer none) The #KiroSb to get the data from
 * Returns: A void pointer the stored data
 * Description:
 *   Calling this function will do the same thing as kiro_sb_get_data, but it
 *   will internaly wait until new data has arived before returning it.
 * Note:
 *   The returned pointer to the element might become invalid at any time by
 *   automatic or manual sync. Under no circumstances might the returned pointer
 *   be freed by the user. If you want to ensure access to the pointed-to data
 *   after a sync, you should use memcpy().
 * See also:
 *   kiro_sb_freeze, kiro_sb_serve, kiro_sb_clone, kiro_sb_push,
 *   kiro_sb_push_dma, kiro_sb_get_data
 */
void*   kiro_sb_get_data_blocking  (KiroSb *sb);

/**
 * kiro_sb_push - Push data into the memory
 * @sb: (transfer none) The #KiroSb to get the data from
 * @data: (transfer none) void pointer to copy data from
 * Returns: %TRUE on success %FALSE in case of error
 * Desciption:
 *   Updates the internal memory by memcopy()-ing the given element into it.
 *   This operation is only valid for 'serving' #KiroSb. Calling this function
 *   on a 'cloning' #KiroSb will allways return %FALSE.
 * Note:
 *   The internal memcopy() will assume an element of the correct size (given
 *   with the initial call to kiro_sb_serve or returned by kiro_sb_get_size) 
 * See also:
 *   kiro_sb_get_size, kiro_sb_serve
 */
gboolean kiro_sb_push       (KiroSb *sb, void *data);

/**
 * kiro_sb_push_dma - Push data into the memory using memory access
 * @sb: (transfer none) The #KiroSb to get the data from
 * Returns: A pointer to the memory where the new data should be stored
 * Desciption:
 *   Returns a pointer where the new data should be stored.
 *   This operation is only valid for a 'serving' #KiroSb. Calling this
 *   function on a 'cloning' #KiroSb will allways return a %NULL pointer.
 * Note:
 *   It is the users responsibility to ensure no more data is written to the
 *   pointed memory then was specified with the initial call to kiro_sb_serve or
 *   returned by kiro_sb_get_size.  Under no circumstances might the returned
 *   pointer be freed by the user.
 * See also:
 *   kiro_sb_get_size, kiro_sb_serve
 */
void* kiro_sb_push_dma      (KiroSb *sb);


G_END_DECLS

#endif //__kiro_sb_H
