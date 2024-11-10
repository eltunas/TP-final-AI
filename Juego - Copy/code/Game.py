
# game.py
import pygame
import sys
from player import Player
import obstacle
from alien import Alien, Extra
from random import choice, randint
from laser import Laser
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

class Game:
    def __init__(self, screen_width, screen_height, renderizar=True):
        self.renderizar = renderizar
        self.screen_width = screen_width
        self.screen_height = screen_height

        pygame.init()
        if self.renderizar:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Space Invaders')
            self.font = pygame.font.Font('../font/Pixeled.ttf', 20)
        else:
            # Crear una ventana mínima para permitir la carga de imágenes
            self.screen = pygame.display.set_mode((1, 1))
            self.font = None

        # Player setup
        player_sprite = Player((self.screen_width / 2, self.screen_height), self.screen_width, 5)
        self.player = pygame.sprite.GroupSingle(player_sprite)

        # Health and score setup
        self.lives = 3
        self.score = 0

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

        # Audio (desactivado durante el entrenamiento)
        if self.renderizar:
            music = pygame.mixer.Sound('../audio/music.wav')
            music.set_volume(0.2)
            music.play(loops=-1)
            self.laser_sound = pygame.mixer.Sound('../audio/laser.wav')
            self.laser_sound.set_volume(0.5)
            self.explosion_sound = pygame.mixer.Sound('../audio/explosion.wav')
            self.explosion_sound.set_volume(0.3)
        else:
            self.laser_sound = None
            self.explosion_sound = None

        # For rewards
        self.last_score = 0
        self.last_lives = 3
        self.initial_lives = 3
        self.esquivo_disparo = False
        self.disparo_fallido = False

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
                break
            elif alien.rect.left <= 0:
                self.alien_direction = 1
                self.alien_move_down(2)
                break

    def alien_move_down(self, distance):
        if self.aliens:
            for alien in self.aliens.sprites():
                alien.rect.y += distance

    def alien_shoot(self):
        if self.aliens.sprites():
            random_alien = choice(self.aliens.sprites())
            laser_sprite = Laser(random_alien.rect.center, 6, self.screen_height)
            self.alien_lasers.add(laser_sprite)
            if self.laser_sound and self.renderizar:
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
                    if self.explosion_sound and self.renderizar:
                        self.explosion_sound.play()
                else:
                    # Si el láser no golpea nada y sale de la pantalla
                    if laser.rect.bottom <= 0:
                        laser.kill()
                        self.disparo_fallido = True

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
                        self.lives = 0
                else:
                    # Si el láser pasa por el jugador sin golpearlo
                    if laser.rect.top >= self.screen_height:
                        laser.kill()
                        self.esquivo_disparo = True

        if self.aliens:
            for alien in self.aliens:
                pygame.sprite.spritecollide(alien, self.blocks, True)

                if pygame.sprite.spritecollide(alien, self.player, False):
                    self.lives = 0  # Terminar el juego

    def display_lives(self):
        if self.renderizar:
            live_surf = pygame.image.load('../graphics/player.png').convert_alpha()
            live_x_start_pos = self.screen_width - (live_surf.get_size()[0] * 2 + 20)
            for live in range(self.lives - 1):
                x = live_x_start_pos + (live * (live_surf.get_size()[0] + 10))
                self.screen.blit(live_surf, (x, 8))

    def display_score(self):
        if self.renderizar:
            score_surf = self.font.render(f'score: {self.score}', False, 'white')
            score_rect = score_surf.get_rect(topleft=(10, -10))
            self.screen.blit(score_surf, score_rect)

    def victory_message(self):
        if self.renderizar and not self.aliens.sprites():
            victory_surf = self.font.render('You won', False, 'white')
            victory_rect = victory_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(victory_surf, victory_rect)

    def run(self):
        if self.renderizar:
            self.screen.fill((0, 0, 0))

            self.player.sprite.lasers.draw(self.screen)
            self.player.draw(self.screen)
            self.blocks.draw(self.screen)
            self.aliens.draw(self.screen)
            self.alien_lasers.draw(self.screen)
            self.extra.draw(self.screen)
            self.display_lives()
            self.display_score()
            self.victory_message()

            pygame.display.flip()

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
            if self.laser_sound and self.renderizar:
                self.laser_sound.play()

        # Actualizar el juego
        self.player.update()
        self.alien_lasers.update()
        self.extra.update()
        self.aliens.update(self.alien_direction)
        self.alien_position_checker()
        self.extra_alien_timer()
        self.collision_checks()

        # Calcular recompensa
        done = self.lives <= 0 or not self.aliens  # Indica si el juego terminó
        state = self.get_game_state()

        # Calcular recompensa utilizando el diccionario
        reward = self.calculate_reward(state, done)

        if self.renderizar:
            self.run()

        # Devolver el nuevo estado, recompensa, indicador de fin
        return state, reward, done

    def calculate_reward(self, state, done):
        reward = 0

        # Recompensa por destruir alienígenas
        current_score = self.score
        if current_score > self.last_score:
            reward += (current_score - self.last_score) * 10
            self.last_score = current_score

        # Penalización por perder vidas
        if self.lives < self.last_lives:
            reward -= 50 * (self.last_lives - self.lives)
            self.last_lives = self.lives

        # Recompensa por esquivar disparos enemigos
        if self.esquivo_disparo:
            reward += 5
            self.esquivo_disparo = False

        # Penalizar disparos fallidos
        if self.disparo_fallido:
            reward -= 1
            self.disparo_fallido = False

        # Recompensa por sobrevivir
        reward += 0.1

        # Recompensa grande por ganar el juego
        if not self.aliens:
            reward += 1000

        # Penalización si el juego termina sin ganar
        if done and self.aliens:
            reward -= 100

        return reward

    def render_game(self):
        # Dibuja todos los sprites en la pantalla
        if self.renderizar:
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

        # Resetear variables adicionales
        self.esquivo_disparo = False
        self.disparo_fallido = False

    def get_closest(self, sprite_group, position):
        """Devuelve el sprite más cercano en el grupo y su distancia a la posición dada."""
        closest_sprite = None
        min_distance = float('inf')

        for sprite in sprite_group.sprites():
            distance = np.linalg.norm(np.array(sprite.rect.center) - np.array(position))
            if distance < min_distance:
                min_distance = distance
                closest_sprite = sprite

        return closest_sprite, min_distance

    def get_game_state(self):
        """Obtiene el estado del juego, incluyendo los elementos más cercanos y la posición del jugador."""
        player_pos = self.player.sprite.rect.center

        # Alien más cercano
        closest_alien, alien_distance = self.get_closest(self.aliens, player_pos)

        # Láser de alien más cercano
        closest_alien_laser, laser_distance = self.get_closest(self.alien_lasers, player_pos)

        # Obstáculo más cercano
        closest_obstacle, obstacle_distance = self.get_closest(self.blocks, player_pos)

        # Crear un diccionario con el estado
        game_state = {
            'player_position': player_pos,
            'closest_alien': (closest_alien.rect.center if closest_alien else None, alien_distance),
            'closest_alien_laser': (closest_alien_laser.rect.center if closest_alien_laser else None, laser_distance),
            'closest_obstacle': (closest_obstacle.rect.center if closest_obstacle else None, obstacle_distance)
        }

        return game_state
