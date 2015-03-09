#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "kiro-server.h"
#include "kiro-trb.h"
#include <gmodule.h>
#include <gio/gio.h>
#include <string.h>
#include <math.h>



static const char g_digits[10][20] = {
    /* 0 */
    {
        0x00, 0xff, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0xff, 0x00, 0x00, 0xff,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0xff, 0xff, 0x00
    },
    /* 1 */
    {
        0x00, 0x00, 0xff, 0x00,
        0x00, 0xff, 0xff, 0x00,
        0x00, 0x00, 0xff, 0x00,
        0x00, 0x00, 0xff, 0x00,
        0x00, 0x00, 0xff, 0x00
    },
    /* 2 */
    {
        0x00, 0xff, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0x00, 0xff, 0x00,
        0x00, 0xff, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff
    },
    /* 3 */
    {
        0x00, 0xff, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0x00, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0xff, 0xff, 0x00
    },
    /* 4 */
    {
        0xff, 0x00, 0x00, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0xff, 0xff, 0xff, 0xff,
        0x00, 0x00, 0x00, 0xff,
        0x00, 0x00, 0x00, 0xff
    },
    /* 5 */
    {
        0xff, 0xff, 0xff, 0xff,
        0xff, 0x00, 0x00, 0x00,
        0x00, 0xff, 0xff, 0x00,
        0x00, 0x00, 0x00, 0xff,
        0xff, 0xff, 0xff, 0x00
    },
    /* 6 */
    {
        0x00, 0xff, 0xff, 0xff,
        0xff, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0xff, 0xff, 0x00
    },
    /* 7 */
    {
        0xff, 0xff, 0xff, 0xff,
        0x00, 0x00, 0x00, 0xff,
        0x00, 0x00, 0xff, 0x00,
        0x00, 0xff, 0x00, 0x00,
        0xff, 0x00, 0x00, 0x00
    },
    /* 8 */
    {
        0x00, 0xff, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0xff, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0xff, 0xff, 0x00
    },
    /* 9 */
    {
        0x00, 0xff, 0xff, 0x00,
        0xff, 0x00, 0x00, 0xff,
        0x00, 0xff, 0xff, 0xff,
        0x00, 0x00, 0x00, 0xff,
        0xff, 0xff, 0xff, 0x00
    }
};

static const guint DIGIT_WIDTH = 4;
static const guint DIGIT_HEIGHT = 5;

static void
print_number (gchar *buffer, guint number, guint x, guint y, guint width)
{
    for (guint i = 0; i < DIGIT_WIDTH; i++) {
        for (guint j = 0; j < DIGIT_HEIGHT; j++) {
            char val = (char) g_digits[number][j * DIGIT_WIDTH + i];

            if (val != 0x00) {
                //This should make the frame counter appear in a bright yellow
                val = 0xBE;
            }

            buffer[ (y + j)*width + (x + i)] = (guint8) val;
        }
    }
}

static void
print_current_frame (gchar *buffer, guint number, guint width, guint height, GRand *rand)
{
    guint divisor = 10000000;
    int x = 1;

    while (divisor > 0) {
        print_number (buffer, number / divisor, x, 1, width);
        number = number % divisor;
        divisor = divisor / 10;
        x += DIGIT_WIDTH + 1;
    }

    //Grayscale pattern is the same for every row. Just calculate one single
    //Scanline, so we can reuse it and dont have to do the whole calculation
    //for every row again.
    char default_line[width];

    for (guint p = 0; p < width; p++) {
        default_line[p] = (char) ((p * 256) / (width));
    }

    //Use memcpy to quickly fill every row with the precalculated grayscale
    //pattern
    for (guint y = 16; y < height; y++) {
        guint index = y * width;
        memcpy (buffer + index, &default_line[0], width);
    }

    //This block will fill a square at the center of the image with normal
    //distributed random data
    const double mean = 128.0;
    const double std = 32.0;

    for (guint y = (height / 3); y < ((height * 2) / 3); y++) {
        guint row_start = y * width;

        for (guint i = (width / 3); i < ((width * 2) / 3); i++) {
            int index = row_start + i;
            double u1 = g_rand_double (rand);
            double u2 = g_rand_double (rand);
            double r = sqrt (-2 * log (u1)) * cos (2 * G_PI * u2);
            buffer[index] = (guint8) (r * std + mean);
        }
    }
}


int 
main (void)
{
    KiroServer *server = kiro_server_new ();
    KiroTrb *rb = kiro_trb_new ();
    kiro_trb_reshape (rb, 512 * 512, 15);
    GRand *rand = g_rand_new();

    if (0 > kiro_server_start (server, NULL, "60010", kiro_trb_get_raw_buffer (rb), kiro_trb_get_raw_size (rb))) {
        g_critical ("Failed to start server properly");
        goto done;
    }

    guint frame = 0;
    gchar *buffer = NULL;

    while (1) {
        sleep (1);
        buffer = kiro_trb_dma_push (rb);
        print_current_frame (buffer, frame, 512, 512, rand);
        frame++;
        if (frame % 1000 == 0)
            kiro_server_realloc (server, kiro_trb_get_raw_buffer (rb), kiro_trb_get_raw_size (rb));
    }

done:
    g_rand_free (rand);
    kiro_trb_free (rb);
    kiro_server_free (server);
    return 0;
}
