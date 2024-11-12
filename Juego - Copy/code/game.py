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

class PlayerGame:
    def __init__(self, screen_width, screen_height, generation, player_id):
        self.player_id = player_id
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize Pygame display to avoid errors
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.init()

        # Player setup
        player_sprite = Player((self.screen_width / 2, self.screen_height), self.screen_width, 5)
        self.player = pygame.sprite.GroupSingle(player_sprite)

        # Health and score setup
        self.lives = 3
        self.score = 0
        self.font = pygame.font.Font('../font/Pixeled.ttf', 20)

        # Obstacle setup
        self.shape = obstacle.shape
        self.block_size = 6
        self.blocks = pygame.sprite.Group()
        self.obstacle_amount = 4
        self.obstacle_x_positions = [num * (self.screen_width / self.obstacle_amount) for num in range(self.obstacle_amount)]
        ##self.create_multiple_obstacles(*self.obstacle_x_positions, x_start=self.screen_width / 15, y_start=480)

        # Alien setup
        self.aliens = pygame.sprite.Group()
        self.alien_lasers = pygame.sprite.Group()
        self.alien_setup(rows=6, cols=8)
        self.alien_direction = 1

        # Extra setup
        self.extra = pygame.sprite.GroupSingle()
        self.extra_spawn_time = randint(40, 80)

        # Recompensas
        self.last_score = 0
        self.last_lives = 3
        self.initial_lives = 3
        self.kills = 0
        self.reach15 = False
        self.reach10 = False
        self.reach5 = False

        # Configuración de temporizador para disparos de aliens
        self.ALIENLASER_EVENT = pygame.USEREVENT + 1
        pygame.time.set_timer(self.ALIENLASER_EVENT, 600)  # Cada 600 ms

        if (generation <= 15):
            # Límite de tiempo para cada jugador
            self.time_limit = 30  # Límite de tiempo en segundos
        elif (generation > 15 and generation <= 30):
            # Límite de tiempo para cada jugador
            self.time_limit = 60  # Límite de tiempo en segundos
        else:
            # Límite de tiempo para cada jugador
            self.time_limit = 120  # Límite de tiempo en segundos

        self.start_time = time.time()  # Guarda el tiempo de inicio

        self.last_life_loss_time = 0

    def check_border_collision(self):
        current_time = time.time()
        
        # Verifica si el jugador está tocando los bordes de la pantalla
        if self.player.sprite.rect.left <= 0 or self.player.sprite.rect.right >= self.screen_width:
            # Comprueba si ha pasado al menos 1 segundo desde la última pérdida de vida
            if current_time - self.last_life_loss_time >= 5:
                self.lives -= 1
                #logging.info(f'Player {self.player_id} hit the border. Remaining lives: {self.lives}')
                self.last_life_loss_time = current_time  # Actualiza el tiempo de la última pérdida de vida


    def run(self):
        # Actualizar lógica del juego
        self.player.update()
        self.alien_lasers.update()
        self.extra.update()
        self.aliens.update(self.alien_direction)
        self.alien_position_checker()
        self.extra_alien_timer()
        self.collision_checks()

        self.check_border_collision()

        # Manejar el evento de disparo de aliens
        for event in pygame.event.get():
            if event.type == self.ALIENLASER_EVENT:
                self.alien_shoot()  # Disparar un láser desde un alien al azar

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
        for row_index in range(rows):
            for col_index in range(cols):
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
        for alien in self.aliens.sprites():
            alien.rect.y += distance

    def alien_shoot(self):
        if self.aliens:
            random_alien = choice(self.aliens.sprites())
            laser_sprite = Laser(random_alien.rect.center, 3, self.screen_height)
            self.alien_lasers.add(laser_sprite)

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
                        self.kills += 1
                        self.score += alien.value * (1 + self.kills/5)
                    laser.kill()

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

        for alien in self.aliens:
            if pygame.sprite.spritecollide(alien, self.blocks, True):
                continue

            if pygame.sprite.spritecollide(alien, self.player, False):
                self.lives = 0

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
        elif action == 4:
            self.player.sprite.move(-1)  # Moverse a la izquierda y dispara
            self.player.sprite.shoot_laser()
        elif action == 5:
            self.player.sprite.move(1)  # Moverse a la derecha, no dispara
        

        self.run()

        # Calcular recompensa
        done = self.lives <= 0  # Indica si el juego terminó
        state = self.get_game_state()

        # Calcular recompensa utilizando el diccionario
        reward = self.calculate_reward(state, done, action)

        # Verificar si el tiempo límite ha sido alcanzado
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        #ARREGLAR PARA MULTIPLES PLAYER
        if elapsed_time >= self.time_limit:
            done = True  # Indica que el juego ha terminado

        if(0 == len(self.aliens)):
            done = True

        
        # Devolver el nuevo estado, recompensa, indicador de fin
        return state, reward, done

    def calculate_reward(self, state, done, action):
        reward = 0

        # Recompensa por disparar
        if action == 3:  # Acción de disparo
            reward += 2  # Motiva a disparar activamente

        # Recompensa por destruir un enemigo (basado en incremento de puntaje)
        current_score = self.score
        if current_score > self.last_score:
            reward += (current_score - self.last_score) # Aumenta el valor por enemigo destruido
            self.last_score = current_score

        # Penalización significativa por perder una vida
        if self.lives < self.last_lives:
            reward -= 1000
            self.last_lives = self.lives

        # Penalización por quedarse alineado con un láser sin moverse
        if state[5] == 1 and action == 0:  # Si hay un láser alineado y el jugador no se mueve
            reward -= 30  # Penalización adicional

        # Recompensa por evadir un láser (movimiento cuando un láser está cerca)
        closest_lasers, distances = self.get_n_closest(self.alien_lasers, self.player.sprite.rect.center, n=1)
        if distances:  # Verificar que haya láseres en la lista
            closest_laser_distance = min(distances)
            if closest_laser_distance < 40 and action in [1, 2, 4, 5]:  # Si un láser está muy cerca y el jugador se mueve
                reward += 10  # Recompensa por evadir con éxito


        #Si esta en el borde recompensarlo si se aleja, penalizarlo si no
        if self.player.sprite.rect.left <= 0: 
            if  action == 1 or action == 4:
                reward += 10
            else:
                reward -= 3
        
        if self.player.sprite.rect.right >= self.screen_width:
            if  action == 2 or action == 5:
                reward += 10
            else:
                reward -= 3

        #Recompensarlo por progreso
        if(len(self.aliens) <= 15 and not self.reach15):
            self.reach15 = True
            reward += 10000

        if(len(self.aliens) <= 10 and not self.reach10):
            self.reach10 = True
            reward += 50000

        if(len(self.aliens) <= 5 and not self.reach5):
            self.reach5 = True
            reward += 100000

        return reward
    
    def render_game(self):
        # Dibuja todos los sprites en la pantalla
        if hasattr(self, 'screen') and self.screen:
            self.screen.fill((0, 0, 0))  # Llenar la pantalla de negro u otro color de fondo
            self.player.draw(self.screen)
            self.blocks.draw(self.screen)
            self.aliens.draw(self.screen)
            self.extra.draw(self.screen)

            # Actualizar la pantalla
            pygame.display.flip()

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
        self.blocks = pygame.sprite.Group()
        ##self.create_multiple_obstacles(*self.obstacle_x_positions, x_start=self.screen_width / 15, y_start=480)

        # Alien setup
        self.aliens = pygame.sprite.Group()
        self.alien_lasers = pygame.sprite.Group()
        self.alien_setup(rows=6, cols=8)
        self.alien_direction = 1

        self.kills = 0

        # Extra setup
        self.extra = pygame.sprite.GroupSingle()
        self.extra_spawn_time = randint(40, 80)
        self.start_time = time.time()

    def get_closest(self, sprite_group, position):
        closest_sprite = None
        min_distance = float('inf')
        
        for sprite in sprite_group.sprites():
            distance = np.linalg.norm(np.array(sprite.rect.center) - np.array(position))
            if distance < min_distance:
                min_distance = distance
                closest_sprite = sprite
        
        return closest_sprite, min_distance

    def get_n_closest(self, sprite_group, position, n=5):
        sprites_with_distance = []

        # Calcular la distancia de cada sprite y almacenarla en una lista
        for sprite in sprite_group.sprites():
            distance = np.linalg.norm(np.array(sprite.rect.center) - np.array(position))
            sprites_with_distance.append((sprite, distance))

        # Ordenar los sprites por distancia y obtener los n más cercanos
        sorted_sprites = sorted(sprites_with_distance, key=lambda x: x[1])[:n]

        # Separar los sprites y las distancias
        closest_sprites = [sprite for sprite, _ in sorted_sprites]
        distances = [distance for _, distance in sorted_sprites]

        return closest_sprites, distances

    def get_game_state(self):
        numCloseAlien = 3
        numCloseLaser = 5
        player_pos = self.player.sprite.rect.center
        player_velocity = self.player.sprite.velocity if hasattr(self.player.sprite, 'velocity') else (0, 0)

        # Distancias a los bordes del juego (normalizadas entre -1 y 1)
        screen_center_x = self.screen_width / 2
        dplayer_relative_x = (self.player.sprite.rect.center[0] - screen_center_x) / screen_center_x  # Normalizado entre -1 y 1

        # Obtener los alienígenas más cercanos y normalizar sus posiciones
        closest_aliens, alien_distances = self.get_n_closest(self.aliens, player_pos, n=numCloseAlien)
        alien_data = []
        for alien, distance in zip(closest_aliens, alien_distances):
            relative_x = (alien.rect.center[0] - player_pos[0]) / self.screen_width  # Normalizado entre -1 y 1
            relative_y = (alien.rect.center[1] - player_pos[1]) / self.screen_height  # Normalizado entre -1 y 1
            alien_velocity_x = (alien.velocity[0] if hasattr(alien, 'velocity') else 0) / 10  # Suponiendo un máximo de velocidad 10

            alien_data.extend([
                relative_x, relative_y, alien_velocity_x
            ])

        # Rellenar con valores por defecto si no hay suficientes alienígenas
        while len(alien_data) < numCloseAlien * 3:
            alien_data.extend([0, 0, 0])

        # Obtener los láseres más cercanos y normalizar sus posiciones relativas
        closest_alien_lasers, laser_distances = self.get_n_closest(self.alien_lasers, player_pos, n=numCloseLaser)

        for laser in closest_alien_lasers:
            laser.set_closest_status(True)

        for laser in self.alien_lasers:
            if laser not in closest_alien_lasers:
                laser.set_closest_status(False)

        laser_data = []
        for laser, distance in zip(closest_alien_lasers, laser_distances):
            relative_x = (laser.rect.center[0] - player_pos[0]) / self.screen_width  # Normalizado entre -1 y 1
            relative_y = (laser.rect.center[1] - player_pos[1]) / self.screen_height  # Normalizado entre -1 y 1
            laser_data.extend([
                relative_y, relative_x
            ])

        # Rellenar con valores por defecto si no hay suficientes láseres
        while len(laser_data) < numCloseLaser * 2:
            laser_data.extend([0, 0])

        # Normalizar la velocidad del jugador en función de un valor máximo esperado (ej. máximo 10)
        player_velocity_x = player_velocity[0] / 10  # Normalizar según la velocidad máxima esperada

        # Indicador de alineación con aliens y láseres
        is_alien_aligned = int(any(alien.rect.centerx == player_pos[0] for alien in self.aliens))
        is_laser_aligned = int(any(laser.rect.centerx == player_pos[0] for laser in self.alien_lasers))

        # Dirección de movimiento de los alienígenas
        alien_direction = self.alien_direction  # Asumiendo -1 para izquierda, 1 para derecha

        # Tiempo desde el último disparo del jugador (normalizado a un máximo de 1 segundo)
        #time_since_last_shot = (time.time() - self.player.sprite.last_shot_time) if hasattr(self.player.sprite, 'last_shot_time') else 1

        # Crear el estado del juego como un vector
        game_state = [
            player_velocity_x,      # Velocidad del jugador normalizada
            dplayer_relative_x,     # Posición relativa en x normalizada
            *alien_data,            # Datos normalizados de los alienígenas más cercanos
            *laser_data,            # Datos normalizados de los láseres más cercanos
            is_alien_aligned,       # Si tiene un alien alineado en x
            is_laser_aligned,       # Si tiene un laser aliendo en X
            alien_direction,        # la direccion de los aliens
            self.lives / self.initial_lives # Las vidas entre 0 y 1
        ]

        return game_state

class Game:
    def __init__(self, screen_width, screen_height, num_players, gen, render=True):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.render = render
        self.num_players = num_players
        self.players = []
        self.clock = pygame.time.Clock()

        # Crear una instancia de PlayerGame para cada jugador
        for i in range(num_players):
            self.players.append(PlayerGame(screen_width, screen_height, gen, player_id=i))

        # Inicializar la pantalla solo si se requiere renderizado
        if self.render:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Space Invaders Multijugador')
        else:
            self.screen = None

    def run(self, background=False):

        # Definir el valor de los FPS según si estamos en background o no
        fps = 60 if not background else 600  # Aumentar significativamente los FPS en background
        self.clock.tick(fps)
        self.screen.fill((0, 0, 0))

        # Actualizar lógica de cada jugador de manera independiente
        for player_game in self.players:
            player_game.run()

        # Renderizar gráficos solo si render es True
        for player_game in self.players:
            player_game.player.sprite.lasers.draw(self.screen)
            player_game.player.draw(self.screen)
            player_game.blocks.draw(self.screen)
            player_game.aliens.draw(self.screen)
            player_game.alien_lasers.draw(self.screen)
            player_game.extra.draw(self.screen)
        pygame.display.flip()

    def reset(self):
        # Resetear todos los jugadores
        for player_game in self.players:
            player_game.reset()