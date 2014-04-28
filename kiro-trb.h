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
 * SECTION: kiro-trb
 * @Short_description: KIRO 'Clever Ring Buffer'
 * @Title: KiroTrb
 *
 * KiroTrb implements a 'Transmittable Ring Buffer' that holds all necessary information
 * about its content inside itself, so its data can be exchanged between different
 * instances of the KiroTrb Class and/or sent over a network.
 */
 
#ifndef __KIRO_TRB_H
#define __KIRO_CBR_H

#include <stdint.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define KIRO_TYPE_TRB             (kiro_trb_get_type())
#define KIRO_TRB(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), KIRO_TYPE_TRB, KiroTrb))
#define KIRO_IS_TRB(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), KIRO_TYPE_TRB))
#define KIRO_TRB_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), KIRO_TYPE_TRB, KiroTrbClass))
#define KIRO_IS_TRB_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), KIRO_TYPE_TRB))
#define KIRO_TRB_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj), KIRO_TYPE_TRB, KiroTrbClass))


typedef struct _KiroTrb           KiroTrb;
typedef struct _KiroTrbClass      KiroTrbClass;
typedef struct _KiroTrbPrivate    KiroTrbPrivate;


struct _KiroTrb {
    
    GObject parent;
    
    /*< private >*/
    KiroTrbPrivate *priv;
};


/**
 * IbvConnectorInterface:
 *
 * Base interface for IbvConnectors.
 */

struct _KiroTrbClass {
    
    GObjectClass parent_class;
       
};


struct KiroTrbInfo {
    
    /* internal information about the buffer */
    uint64_t buffer_size_bytes;  // Size in bytes INCLUDING this header
    uint64_t element_size;       // Size in bytes of one single element
    uint64_t offset;             // Current Offset to access the 'oldest' element (in element count!)
    
} __attribute__((packed));


/* GObject and GType functions */
GType       kiro_trb_get_type           (void);

GObject     kiro_trb_new                (void);

/* trb functions */

uint64_t    kiro_trb_get_element_count  (KiroTrb* trb);

uint64_t    kiro_trb_get_element_size   (KiroTrb* trb);

uint64_t    kiro_trb_get_max_elements   (KiroTrb* trb);

uint64_t    kiro_trb_get_raw_size       (KiroTrb* trb);


/**
 * kiro_trb_get_raw_buffer - Returns a pointer to the buffer memory
 * @self: KIRO TRB to perform the operation on
 * Description:
 *   Returns a pointer to the memory structure of the given buffer.
 * Notes:
 *   The returned pointer points to the beginning of the internal
 *   memory of the buffer, including all header information. The
 *   user is responsible to ensure the consistency of any data
 *   written to the memory and should call 'kiro_trb_refesh' in
 *   case any header information was changed.
 *   The pointed to memory might become invalid at any time by
 *   concurrent access to the TRB, reshaping, adopting or cloning
 *   a new memory block.
 *   Under no circumstances might the memory pointed to by the returned
 *   pointer be 'freed' by the user!
 * See also:
 *   kiro_trb_refesh, kiro_trb_reshape, kiro_trb_adopt, kiro_trb_clone
 */
void*       kiro_trb_get_raw_buffer     (KiroTrb* trb);


/**
 * kiro_trb_get_element - Returns a pointer to the element at the given
 * index.
 * @self: KIRO TRB to perform the operation on
 * @index: Index of the element in the buffer to access
 * Description:
 *   Returns a pointer to the element in the buffer at the given index.
 * Notes:
 *   The returned pointer to the element is only guaranteed to be valid
 *   immediately after the function call. The user is responsible to
 *   ensure that no data is written to the returned memory. The
 *   element pointed to might become invalid at any time by any concurrent
 *   access to the buffer wraping around and overwriting the element or
 *   changing the buffer memory entirely.
 *   Under no circumstances might the memory pointed to by the returned
 *   pointer be 'freed' by the user!
 * See also:
 *   kiro_trb_get_element_size, kiro_trb_get_raw_buffer
 */
void*       kiro_trb_get_element        (KiroTrb* trb, uint64_t index);


/**
 * kiro_trb_dma_push - Gives DMA to the next element and pushes the buffer
 * @self: KIRO TRB to perform the operation on
 * Description:
 *   Returns a pointer to the next element in the buffer and increases
 *   all internal counters and meta data as if an element was pushed
 *   onto the buffer.
 * Notes:
 *   The returned pointer to the element is only guaranteed to be valid
 *   immediately after the function call. The user is responsible to
 *   ensure that no more data is written than 'element_size'. The
 *   element pointed to might become invalid at any time by any concurrent
 *   access to the buffer wraping around and overwriting the element or
 *   changing the buffer memory entirely.
 *   Under no circumstances might the memory pointed to by the returned
 *   pointer be 'freed' by the user!
 * See also:
 *   kiro_trb_push, kiro_trb_get_element_size, kiro_trb_get_raw_buffer
 */
void*       kiro_trb_dma_push           (KiroTrb*);


void        kiro_trb_flush              (KiroTrb* trb);

int         kiro_trb_is_setup           (KiroTrb* trb);

int         kiro_trb_reshape            (KiroTrb* trb, uint64_t element_size, uint64_t element_count);

int         kiro_trb_clone              (KiroTrb* trb, void* source);

int         kiro_trb_push               (KiroTrb* trb, void* source);

void        kiro_trb_refresh            (KiroTrb* trb);

void        kiro_trb_adopt              (KiroTrb* trb, void* source);

G_END_DECLS

#endif //__KIRO_TRB_H