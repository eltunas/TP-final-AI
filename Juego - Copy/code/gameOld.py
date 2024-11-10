import pygame
from player import Player
import obstacle
from alien import Alien, Extra
from random import choice, randint
from laser import Laser
import numpy as np
import time
import tensorflow as tf
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


class Game:
    def __init__(self, screen_width, screen_height):

        
        self.screen_width = screen_width  # Almacenar como atributo
        self.screen_height = screen_height  # Almacenar como atributo

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Space Invaders')

        # Player setup
        player_sprite = Player((self.screen_width / 2, self.screen_height), self.screen_width, 5)
        self.player = pygame.sprite.GroupSingle(player_sprite)

        # Health and score setup
        self.lives = 3
        self.live_surf = pygame.image.load('../graphics/player.png').convert_alpha()
        self.live_x_start_pos = self.screen_width - (self.live_surf.get_size()[0] * 2 + 20)
        self.score = 0
        self.font = pygame.font.Font('../font/Pixeled.ttf', 20)

        # Obstacle setup
        self.shape = obstacle.shape
        self.block_size = 6
        self.blocks = pygame.sprite.Group()
        self.obstacle_amount = 4
        self.obstacle_x_positions = [num * (self.screen_width / self.obstacle_amount) for num in range(self.obstacle_amount)]
        self.create_multiple_obstacles(*self.obstacle_x_positions, x_start=self.screen_width / 15, y_start=480)

        # Alien setup
        self.aliens = pygame.sprite.Group()
        self.alien_lasers = pygame.sprite.Group()
        self.alien_setup(rows=6, cols=8)
        self.alien_direction = 1

        # Extra setup
        self.extra = pygame.sprite.GroupSingle()
        self.extra_spawn_time = randint(40, 80)

        # Audio
        music = pygame.mixer.Sound('../audio/music.wav')
        music.set_volume(0.2)
        music.play(loops=-1)
        self.laser_sound = pygame.mixer.Sound('../audio/laser.wav')
        self.laser_sound.set_volume(0.5)
        self.explosion_sound = pygame.mixer.Sound('../audio/explosion.wav')
        self.explosion_sound.set_volume(0.3)

        #For rewards
        self.last_score = 0
        self.last_lives = 3
        self.initial_lives = 3

    def create_obstacle(self, x_start, y_start, offset_x):
        for row_index, row in enumerate(self.shape):
            for col_index, col in enumerate(row):
                if col == 'x':
                    x = x_start + col_index * self.block_size + offset_x
                    y = y_start + row_index * self.block_size
                    block = obstacle.Block(self.block_size, (241, 79, 80), x, y)
                    self.blocks.add(block)

    def create_multiple_obstacles(self, *offset, x_start, y_start):
        for offset_x in offset:
            self.create_obstacle(x_start, y_start, offset_x)

    def alien_setup(self, rows, cols, x_distance=60, y_distance=48, x_offset=70, y_offset=100):
        for row_index, row in enumerate(range(rows)):
            for col_index, col in enumerate(range(cols)):
                x = col_index * x_distance + x_offset
                y = row_index * y_distance + y_offset

                if row_index == 0:
                    alien_sprite = Alien('yellow', x, y)
                elif 1 <= row_index <= 2:
                    alien_sprite = Alien('green', x, y)
                else:
                    alien_sprite = Alien('red', x, y)
                self.aliens.add(alien_sprite)

    def alien_position_checker(self):
        all_aliens = self.aliens.sprites()
        for alien in all_aliens:
            if alien.rect.right >= self.screen_width:
                self.alien_direction = -1
                self.alien_move_down(2)
            elif alien.rect.left <= 0:
                self.alien_direction = 1
                self.alien_move_down(2)

    def alien_move_down(self, distance):
        if self.aliens:
            for alien in self.aliens.sprites():
                alien.rect.y += distance

    def alien_shoot(self):
        if self.aliens.sprites():
            random_alien = choice(self.aliens.sprites())
            laser_sprite = Laser(random_alien.rect.center, 6, self.screen_height)
            self.alien_lasers.add(laser_sprite)
            self.laser_sound.play()

    def extra_alien_timer(self):
        self.extra_spawn_time -= 1
        if self.extra_spawn_time <= 0:
            self.extra.add(Extra(choice(['right', 'left']), self.screen_width))
            self.extra_spawn_time = randint(400, 800)

    def collision_checks(self):
        if self.player.sprite.lasers:
            for laser in self.player.sprite.lasers:
                if pygame.sprite.spritecollide(laser, self.blocks, True):
                    laser.kill()

                aliens_hit = pygame.sprite.spritecollide(laser, self.aliens, True)
                if aliens_hit:
                    for alien in aliens_hit:
                        self.score += alien.value
                    laser.kill()
                    self.explosion_sound.play()

                if pygame.sprite.spritecollide(laser, self.extra, True):
                    self.score += 500
                    laser.kill()

        if self.alien_lasers:
            for laser in self.alien_lasers:
                if pygame.sprite.spritecollide(laser, self.blocks, True):
                    laser.kill()

                if pygame.sprite.spritecollide(laser, self.player, False):
                    laser.kill()
                    self.lives -= 1
                    if self.lives <= 0:
                        pygame.quit()
                        sys.exit()

        if self.aliens:
            for alien in self.aliens:
                pygame.sprite.spritecollide(alien, self.blocks, True)

                if pygame.sprite.spritecollide(alien, self.player, False):
                    pygame.quit()
                    sys.exit()

    def display_lives(self):
        for live in range(self.lives - 1):
            x = self.live_x_start_pos + (live * (self.live_surf.get_size()[0] + 10))
            self.screen.blit(self.live_surf, (x, 8))

    def display_score(self):
        score_surf = self.font.render(f'score: {self.score}', False, 'white')
        score_rect = score_surf.get_rect(topleft=(10, -10))
        self.screen.blit(score_surf, score_rect)

    def victory_message(self):
        if not self.aliens.sprites():
            victory_surf = self.font.render('You won', False, 'white')
            victory_rect = victory_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(victory_surf, victory_rect)

    def run(self):
        self.screen.fill((0, 0, 0))
        
        self.player.update()
        self.alien_lasers.update()
        self.extra.update()
        self.aliens.update(self.alien_direction)
        self.alien_position_checker()
        self.extra_alien_timer()
        self.collision_checks()

        self.player.sprite.lasers.draw(self.screen)
        self.player.draw(self.screen)
        self.blocks.draw(self.screen)
        self.aliens.draw(self.screen)
        self.alien_lasers.draw(self.screen)
        self.extra.draw(self.screen)
        self.display_lives()
        self.display_score()
        self.victory_message()

    def step(self, action):
        # Ejecutar la acción
        if action == 1:
            self.player.sprite.move(-1)
        elif action == 2:
            self.player.sprite.move(1)
        elif action == 3 and self.player.sprite.ready:
            self.player.sprite.shoot_laser()
            self.player.sprite.ready = False
            self.player.sprite.laser_time = pygame.time.get_ticks()
            self.player.sprite.laser_sound.play()

        self.run()

        # Calcular recompensa
        done = self.lives <= 0  # Indica si el juego terminó

        # Capturar el estado del juego
        state = self.get_state_image()  # Ahora devuelve un diccionario
        state = np.expand_dims(state, axis=0)  # Agregar una dimensión para el batch

        # Calcular recompensa utilizando el diccionario
        reward = self.calculate_reward(state, done)
        
        # Devolver el nuevo estado, recompensa, indicador de fin
        return state, reward, done

    def calculate_reward(self, state, done):
        reward = 0

        # Recompensa por destruir un enemigo (basado en incremento de puntaje)
        current_score = self.score
        if current_score > self.last_score:
            reward += (current_score - self.last_score)  # Ajusta el multiplicador según sea necesario
            self.last_score = current_score

        if self.lives < self.last_lives:
            reward -= 1  # Ajusta la penalización si se pierde una vida
            self.last_lives = self.lives

        if done:
            reward -= 5  # Penalización adicional si se pierde el juego

        return reward
    
    def render_game(self):
        # Dibuja todos los sprites en la pantalla
        self.screen.fill((0, 0, 0))  # Llenar la pantalla de negro u otro color de fondo
        self.player.draw(self.screen)
        self.blocks.draw(self.screen)
        self.aliens.draw(self.screen)
        self.extra.draw(self.screen)

        # Actualizar la pantalla
        pygame.display.flip()

        # Devolver la superficie del juego
        return self.screen  # Retorna la superficie sin convertirla a numpy
    
    def reset(self):
        # Resetear el jugador
        player_sprite = Player((self.screen_width / 2, self.screen_height), self.screen_width, 5)
        self.player = pygame.sprite.GroupSingle(player_sprite)

        # Resetear puntaje y vidas
        self.lives = self.initial_lives
        self.score = 0
        self.last_score = 0
        self.last_lives = self.initial_lives

        # Obstacle setup
        self.shape = obstacle.shape
        self.block_size = 6
        self.blocks = pygame.sprite.Group()
        self.obstacle_amount = 4
        self.obstacle_x_positions = [num * (self.screen_width / self.obstacle_amount) for num in range(self.obstacle_amount)]
        self.create_multiple_obstacles(*self.obstacle_x_positions, x_start=self.screen_width / 15, y_start=480)

        # Alien setup
        self.aliens = pygame.sprite.Group()
        self.alien_lasers = pygame.sprite.Group()
        self.alien_setup(rows=6, cols=8)
        self.alien_direction = 1

        # Extra setup
        self.extra = pygame.sprite.GroupSingle()
        self.extra_spawn_time = randint(40, 80)

        # Obtener la imagen inicial del estado del juego
        initial_image = self.get_state_image()  # Esto ahora devolverá solo el arreglo de numpy

        # Devolver como un diccionario
        return {'image': initial_image}  # Esto es correcto y debería funcionar