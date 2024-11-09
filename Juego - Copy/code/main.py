import pygame, sys
from player import Player
import obstacle
from alien import Alien, Extra
from random import choice, randint
from laser import Laser
import os
import numpy as np
from game import Game

from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


if __name__ == '__main__':
    pygame.init()
    screen_width = 800
    screen_height = 600
    #game = Game(screen_width, screen_height)
    game = Game(screen_width, screen_height, 4, render=True)
    clock = pygame.time.Clock()

    ALIENLASER = pygame.USEREVENT + 1
    pygame.time.set_timer(ALIENLASER, 800)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            #if event.type == ALIENLASER:
                #game.alien_shoot()

        #game.screen.fill((30, 30, 30))
        game.run()

        pygame.display.flip()
        clock.tick(60)
