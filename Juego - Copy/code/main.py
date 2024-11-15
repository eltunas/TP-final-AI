import pygame, sys
from player import Player
import obstacle
from alien import Alien, Extra
from random import choice, randint
from laser import Laser
import os
import numpy as np
import tensorflow as tf
import time

os.chdir(os.path.dirname(__file__))

class Game:
    def __init__(self, screen_width, screen_height, render=True):

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.render = render  # Controla si se renderiza la ventana o no
        self.time_limit = 20  # Límite de tiempo en segundos
        self.start_time = time.time()  # Guarda el tiempo de inicio

        # Inicializar la pantalla solo si render está habilitado
        if self.render:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Space Invaders')
        else:
            self.screen = None  # No se inicializa la pantalla si no se renderiz

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
                        done = True

        if self.aliens:
            for alien in self.aliens:
                pygame.sprite.spritecollide(alien, self.blocks, True)

                if pygame.sprite.spritecollide(alien, self.player, False):
                    self.lives = 0
                    done = True

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
            done = True
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

    def step(self, action, start_time):
        # Ejecutar la acción
        if action == 1:
            self.player.sprite.move(-1)
        elif action == 2:
            self.player.sprite.move(1)
        elif action == 3 and self.player.sprite.ready:
            self.player.sprite.shoot_laser()
            self.player.sprite.ready = False
            self.player.sprite.laser_time = pygame.time.get_ticks()
            #self.player.sprite.laser_sound.play()

        self.run()

        # Calcular recompensa
        done = self.lives <= 0  # Indica si el juego terminó
        state = self.get_game_state()

        # Calcular recompensa utilizando el diccionario
        reward = self.calculate_reward(state, done, start_time)
        
        # Devolver el nuevo estado, recompensa, indicador de fin
        return state, reward, done
    

    def calculate_reward(self, state, done, start_time):
        reward = 0

        # Recompensa por destruir un enemigo (basado en incremento de puntaje)
        current_score = self.score
        if current_score > self.last_score:
            reward += (current_score - self.last_score) 
            self.last_score = current_score

        if self.lives < self.last_lives:
            reward -= 1000  
            print (f"-1 Vida. Te quedan {self.lives}")
            self.last_lives = self.lives

        if self.lives == 0:
            reward -= 500  # Penalización adicional si se pierde el juego
            print("Muerto")

        if not self.aliens.sprites():
            reward += 2000 - (time.time() - start_time)
        
        # Verificar si el jugador está tocando los bordes de la pantalla
        if self.player.sprite.rect.left <= 0:
            reward -= 50
        elif self.player.sprite.rect.right >= self.screen_width:
            reward -= 50

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

    def get_closest_n(self, sprite_group, position, n=5):
        """Devuelve las n posiciones relativas en el eje x más cercanas en el grupo, normalizadas por el ancho de la pantalla."""
        distances = []

        for sprite in sprite_group.sprites():
            # Calcula la distancia en el eje x y normaliza
            distance_x = (sprite.rect.centerx - position[0]) / self.screen_width
            distances.append((sprite, abs(distance_x)))  # Guarda la distancia absoluta y el sprite

        # Ordena por la distancia en el eje x y selecciona los n más cercanos
        closest_sprites = sorted(distances, key=lambda x: x[1])[:n]

        # Devuelve solo las posiciones relativas normalizadas en el eje x
        closest_x_distances = [(sprite.rect.centerx - position[0]) / self.screen_width for sprite, _ in closest_sprites]

        # Si hay menos de n objetos, rellena con valores por defecto (por ejemplo, 0.0)
        while len(closest_x_distances) < n:
            closest_x_distances.append(0.0)  # Valor neutro o muy alto para indicar ausencia

        return closest_x_distances


    def get_game_state(self):
        """Obtiene el estado del juego, incluyendo la posición del jugador y las posiciones relativas de enemigos y láseres."""
        
        # 1. Obtener la posición normalizada del jugador en el eje x
        player_pos = self.player.sprite.rect.center
        player_x_normalized = player_pos[0] / self.screen_width  # Normaliza en el eje x

        # 2. Obtener las posiciones relativas de los 5 aliens más cercanos en el eje x
        alien_x_distances = self.get_closest_n(self.aliens, player_pos, n=5)

        # 3. Obtener las posiciones relativas de los 5 láseres enemigos más cercanos en el eje x
        enemy_laser_x_distances = self.get_closest_n(self.alien_lasers, player_pos, n=5)

        # 4. Direccion general de los enemigos en el eje x (aproximación simple)
        enemy_direction = 0.5 if any(sprite.rect.centerx > player_pos[0] for sprite in self.aliens.sprites()) else -0.5

        # Crear el diccionario de estado del juego
        game_state = {
            'player_position': player_x_normalized,        # Posición normalizada del jugador en el eje x
            'alien_x_distances': alien_x_distances,        # Posiciones relativas en el eje x de los aliens más cercanos
            'enemy_laser_x_distances': enemy_laser_x_distances,  # Posiciones relativas en el eje x de los láseres enemigos
            'enemy_direction': enemy_direction             # Dirección general de los enemigos en el eje x
        }
        
        return game_state

