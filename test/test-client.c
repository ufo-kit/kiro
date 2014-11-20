#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "kiro-client.h"
#include "kiro-trb.h"
#include <SDL/SDL.h>
#include <assert.h>


static _Bool 
init_app (const char *name, SDL_Surface *icon, uint32_t flags)
{
    atexit (SDL_Quit);

    if (SDL_Init (flags) < 0)
        return 0;

    SDL_WM_SetCaption (name, name);
    SDL_WM_SetIcon (icon, NULL);
    return 1;
}

static void 
render (SDL_Surface *sf)
{
    SDL_Surface *screen = SDL_GetVideoSurface();

    if (SDL_BlitSurface (sf, NULL, screen, NULL) == 0)
        SDL_UpdateRect (screen, 0, 0, 0, 0);
}

static int 
filter (const SDL_Event *event)
{
    return event->type == SDL_QUIT;
}


int 
main ( int argc, char *argv[] )
{
    if (argc < 3) {
        printf ("Not enough aruments. Usage: 'kiro-client-sdl <address> <port>'\n");
        return -1;
    }

    KiroClient *client = kiro_client_new ();
    KiroTrb *trb = kiro_trb_new ();

    if (-1 == kiro_client_connect (client, argv[1], argv[2])) {
        kiro_client_free (client);
        return -1;
    }

    kiro_client_sync (client);
    kiro_trb_adopt (trb, kiro_client_get_memory (client));
    
    
    _Bool ok =
        init_app ("UCA Images", NULL, SDL_INIT_VIDEO) &&
        SDL_SetVideoMode (512, 512, 8, SDL_HWSURFACE);
    assert (ok);
    uint32_t mask = 0xffffffff;
    SDL_Surface *data_sf = SDL_CreateRGBSurfaceFrom (
                               kiro_trb_get_element (trb, 0), 512, 512, 8, 512,
                               mask, mask, mask, 0);
   

    SDL_Color colors[256];
    for (int i = 0; i < 256; i++) {
        colors[i].r = i;
        colors[i].g = i;
        colors[i].b = i;
    }
    SDL_SetPalette (data_sf, SDL_LOGPAL | SDL_PHYSPAL, colors, 0, 256);
    SDL_SetEventFilter (filter);
    
    
    int cont = 1;
    while (cont) {
        for (SDL_Event event; SDL_PollEvent (&event);)
            if (event.type == SDL_QUIT) cont = 0;

        if (kiro_client_sync (client) < 0) {
            g_warning ("Unable to get data from server. Stopping.");
            break;
        }
        SDL_Delay (10);
        render (data_sf);
    }

    kiro_client_free (client);
    kiro_trb_free (trb);
    return 0;
}








