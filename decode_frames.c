#include "SDL2/SDL.h"
#include <stdio.h>
#include <assert.h>

#define WIDTH 320
#define HEIGHT 240
static uint8_t global_pixels[WIDTH*HEIGHT*3];

int main(void)
{
	int32_t status = SDL_Init(SDL_INIT_VIDEO);
	assert(status == 0);

	SDL_Window *window = SDL_CreateWindow("test",
					      0,
					      0,
					      WIDTH,
					      HEIGHT,
					      SDL_WINDOW_RESIZABLE);
	assert(window != NULL);

	SDL_Renderer *renderer = SDL_CreateRenderer(window,
						    -1,
						    SDL_RENDERER_SOFTWARE);
	assert(renderer != NULL);

	FILE *stream = popen("ffmpeg -i temp.mp4 -pix_fmt rgb24 -vf fps=1 -c:v rawvideo -map 0:v -f rawvideo pipe:1", "r");
	assert(stream != NULL);

	while (feof(stream) == 0) {
		uint32_t total_count = 0;
		uint32_t count = 0;
		do {
			count = fread(global_pixels + total_count,
				      1,
				      WIDTH,
				      stream);
			total_count += count;
		} while ((count > 0) && (total_count < sizeof(global_pixels)));

		SDL_Surface *surface = SDL_CreateRGBSurfaceFrom(global_pixels,
								WIDTH,
								HEIGHT,
								24,
								3*WIDTH,
								0xFF,
								0xFF,
								0xFF,
								0x00);
		assert(surface != NULL);

		SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
		assert(texture != NULL);

		SDL_FreeSurface(surface);

		status = SDL_RenderCopy(renderer,
					texture,
					NULL,
					NULL);
		assert(status == 0);

		SDL_RenderPresent(renderer);

		SDL_Delay(1000);

		SDL_DestroyTexture(texture);
	}

	status = pclose(stream);
	assert(status != -1);

	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return EXIT_SUCCESS;
}
