#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "kiro-trb.h"

struct test {
    uint32_t zahl;
    uint8_t buchstabe;
} __attribute__ ((packed));


int main (void)
{
    /*
    void* ptr = malloc(sizeof(struct test) + sizeof(uint64_t));
    memset(ptr, 0xFA, sizeof(struct test) + sizeof(uint64_t));
    struct test foo;
    foo.zahl = 42;
    foo.buchstabe = 'R';
    memcpy(ptr, &foo, sizeof(foo));

    struct test *tmp = (struct test *)ptr;
    printf("Zahl = %d\n",tmp->zahl);
    printf("Buchstabe = %c\n", tmp->buchstabe);
    printf("Remaining = %x\n", *((uint64_t *)(ptr+sizeof(struct test))));
    */
    KiroTrb *rb = g_object_new (KIRO_TYPE_TRB, NULL);
    kiro_trb_reshape (rb, sizeof (uint64_t), 3);
    void *buffer = kiro_trb_get_raw_buffer (rb);
    uint64_t foo = 0xAFFED00F;
    uint64_t bar = 0x1337BEEF;
    memcpy (kiro_trb_dma_push (rb), &foo, sizeof (foo));
    memcpy (kiro_trb_dma_push (rb), &foo, sizeof (foo));
    memcpy (kiro_trb_dma_push (rb), &foo, sizeof (foo));
    kiro_trb_push (rb, &bar);
    kiro_trb_push (rb, &foo);
    kiro_trb_push (rb, &foo);
    uint64_t *maman = kiro_trb_get_element (rb, 3);
    printf ("Stored in old: %x\n", *maman);
    KiroTrb *rb2 = g_object_new (KIRO_TYPE_TRB, NULL);
    kiro_trb_clone (rb2, kiro_trb_get_raw_buffer (rb));
    maman = kiro_trb_get_element (rb2, 3);
    printf ("Stored in New: %x\n", *maman);
    sleep (1);
    g_object_unref (rb);
    g_object_unref (rb2);
    return 0;
}