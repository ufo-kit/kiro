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
 * kiro_sb_serve - Allow remote KiroSbs to clone this buffers memory
 * Returns: A gboolean. TRUE = success. FALSE = fail.
 * @sb: (transfer none): The #KiroSb to perform this operation on
 * @size: Size in bytes of the content that will be served
 * Description:
 *   Allows other remote #KiroSbs to connect to this #KiroSb and clone its
 *   memory.
 * Note:
 *   A #KiroSb that already 'serves' its content can no longer clone
 *   other remote #KiroSbs.
 * See also:
 *
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

G_END_DECLS

#endif //__kiro_sb_H
