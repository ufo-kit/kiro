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
 * @Short_description: KIRO 'Transmittable Ring Buffer'
 * @Title: KiroTrb
 *
 * KiroTrb implements a 'Transmittable Ring Buffer' that holds all necessary information
 * about its content inside itself, so its data can be exchanged between different
 * instances of the KiroTrb Class and/or sent over a network.
 */

#include <stdio.h>
 
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include "kiro-trb.h"


/*
 * Definition of 'private' structures and members and macro to access them
 */

#define KIRO_TRB_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), KIRO_TYPE_TRB, KiroTrbPrivate))

struct _KiroTrbPrivate {

    /* Properties */
    // PLACEHOLDER //

    /* 'Real' private structures */
    /* (Not accessible by properties) */
    int         initialized;    // 1 if Buffer is Valid, 0 otherwise
    void*       mem;            // Access to the actual buffer in Memory
    void*       frame_top;      // First byte of the buffer storage
    void*       current;        // Pointer to the current fill state
    uint64_t    element_size;   
    uint64_t    max_elements;
    uint64_t    iteration;      // How many times the buffer has wraped around
    
    /* easy access */
    uint64_t    buff_size;
};


G_DEFINE_TYPE_WITH_PRIVATE (KiroTrb, kiro_trb, G_TYPE_OBJECT);


static
void kiro_trb_init (KiroTrb *self)
{
    KiroTrbPrivate *priv = KIRO_TRB_GET_PRIVATE(self);
    priv->initialized = 0;
}

static void
kiro_trb_finalize (GObject *object)
{
    KiroTrb *self = KIRO_TRB(object);
    KiroTrbPrivate *priv = KIRO_TRB_GET_PRIVATE(self);
    if(priv->mem)
        free(priv->mem);
}

static void
kiro_trb_class_init (KiroTrbClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    gobject_class->finalize = kiro_trb_finalize;
}


/* Privat functions */

void write_header (KiroTrbPrivate* priv)
{
    if(!priv)
        return;
    struct KiroTrbInfo* tmp_info = (struct KiroTrbInfo*)priv->mem;
    tmp_info->buffer_size_bytes = priv->buff_size;
    tmp_info->element_size = priv->element_size;
    tmp_info->offset = (priv->iteration * priv->max_elements) + ((priv->current - priv->frame_top) / priv->element_size);
    memcpy(priv->mem, tmp_info, sizeof(struct KiroTrbInfo));
}



/* TRB functions */

uint64_t kiro_trb_get_element_size (KiroTrb* self)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self);
    if(priv->initialized != 1)
        return 0;
    return priv->element_size;
}


uint64_t kiro_trb_get_max_elements (KiroTrb* self)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self);
    if(priv->initialized != 1)
        return 0;
    return priv->max_elements;
}


uint64_t kiro_trb_get_raw_size (KiroTrb* self)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self);
    if(priv->initialized != 1)
        return 0;
    return priv->buff_size;
}


void* kiro_trb_get_raw_buffer (KiroTrb* self)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self); 
    if(priv->initialized != 1)
        return NULL;
    write_header(priv);
    return priv->mem;
}



void* kiro_trb_get_element (KiroTrb* self, uint64_t element)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self);
    if(priv->initialized != 1)
        return NULL;
    
    uint64_t relative = 0;    
    if(priv->iteration == 0)
        relative = element * priv->element_size;
    else
        relative = ((priv->current - priv->frame_top) + (priv->element_size * element)) % (priv->buff_size - sizeof(struct KiroTrbInfo));
        
    return priv->frame_top + relative;
}


void kiro_trb_flush (KiroTrb *self)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self);
    priv->iteration = 0;
    priv->current = priv->frame_top;
}


int kiro_trb_reshape (KiroTrb *self, uint64_t element_size, uint64_t element_count)
{
    size_t new_size = (element_size * element_count) + sizeof(struct KiroTrbInfo);
    void* newmem = malloc(new_size);
    if(!newmem)
        return -1;
    ((struct KiroTrbInfo *)newmem)->buffer_size_bytes = new_size;
    ((struct KiroTrbInfo *)newmem)->element_size = element_size;
    ((struct KiroTrbInfo *)newmem)->offset = 0;
    kiro_trb_ingest(self, newmem);
    return 0;
}


int kiro_trb_push (KiroTrb *self, void *element_in)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self);
    if(priv->initialized != 1)
        return -1;
    if((priv->current + priv->element_size) > (priv->mem + priv->buff_size))
        return -1;
    memcpy(priv->current, element_in, priv->element_size);
    priv->current += priv->element_size;
    if(priv->current >= priv->frame_top + (priv->element_size * priv->max_elements))
    {
        priv->current = priv->frame_top;
        priv->iteration++;
    }
    return 0;        
}


void kiro_trb_ingest (KiroTrb *self, void *buff_in)
{
    KiroTrbPrivate* priv = KIRO_TRB_GET_PRIVATE(self);
    struct KiroTrbInfo *tmp = (struct KiroTrbInfo *)buff_in;
    if(priv->mem)
        free(priv->mem);
    priv->mem = buff_in;
    priv->buff_size = tmp->buffer_size_bytes;
    priv->element_size = tmp->element_size;
    priv->max_elements = (tmp->buffer_size_bytes - sizeof(struct KiroTrbInfo)) / tmp->element_size;
    priv->iteration = tmp->offset / priv->max_elements;
    priv->frame_top = buff_in + sizeof(struct KiroTrbInfo);
    priv->current = priv->frame_top + ((tmp->offset % priv->max_elements) * priv->element_size);
    priv->initialized = 1;
}