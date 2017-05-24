#include "SDL2/SDL.h"
#include <stdio.h>
#include <assert.h>

int main(void)
{
	int32_t status = SDL_Init(SDL_INIT_VIDEO);
	assert(status == 0);

	uint32_t img_dim = 32;
	SDL_Window *window = SDL_CreateWindow("test",
					      0,
					      0,
					      img_dim,
					      32,
					      SDL_WINDOW_RESIZABLE);
	assert(window != NULL);

	uint8_t pixels[img_dim*img_dim*3];

	SDL_Surface *surface = SDL_CreateRGBSurfaceFrom(pixels,
							img_dim,
							img_dim,
							24,
							3*img_dim,
							0xFF,
							0xFF,
							0xFF,
							0x00);
	assert(surface != NULL);

	SDL_Renderer *renderer = SDL_CreateRenderer(window,
						    -1,
						    SDL_RENDERER_ACCELERATED);
	assert(renderer != NULL);

	SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
	assert(texture != NULL);

	SDL_FreeSurface(surface);

	status = SDL_RenderCopy(renderer,
				texture,
				NULL,
				NULL);
	assert(status == 0);

	SDL_Delay(3000);

	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return EXIT_SUCCESS;
}
