import pygame
from player import Player
import obstacle
from alien import Alien, Extra
from random import choice, randint
from laser import Laser
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

class PlayerGame:
    def __init__(self, screen_width, screen_height, player_id):
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
        self.lives = 2
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

        # Configuración de temporizador para disparos de aliens
        self.ALIENLASER_EVENT = pygame.USEREVENT + 1
        pygame.time.set_timer(self.ALIENLASER_EVENT, 600)  # Cada 600 ms

        # Límite de tiempo para cada jugador
        self.time_limit = 45  # Límite de tiempo en segundos
        self.start_time = time.time()  # Guarda el tiempo de inicio

    def run(self):
        # Actualizar lógica del juego
        self.player.update()
        self.alien_lasers.update()
        self.extra.update()
        self.aliens.update(self.alien_direction)
        self.alien_position_checker()
        self.extra_alien_timer()
        self.collision_checks()

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
                        logging.info(f'Player {self.player_id} destroyed an alien. Current score: {self.score}')
                    laser.kill()

                if pygame.sprite.spritecollide(laser, self.extra, True):
                    self.score += 500
                    logging.info(f'Player {self.player_id} destroyed an extra alien. Current score: {self.score}')
                    laser.kill()

        if self.alien_lasers:
            for laser in self.alien_lasers:
                if pygame.sprite.spritecollide(laser, self.blocks, True):
                    laser.kill()

                if pygame.sprite.spritecollide(laser, self.player, False):
                    laser.kill()
                    self.lives -= 1
                    logging.info(f'Player {self.player_id} was hit. Remaining lives: {self.lives}')

        for alien in self.aliens:
            if pygame.sprite.spritecollide(alien, self.blocks, True):
                continue

            if pygame.sprite.spritecollide(alien, self.player, False):
                self.lives = 0
                logging.info(f'Player {self.player_id} collided with an alien and lost all lives.')

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

        self.run()

        # Verificar si el jugador está tocando los bordes de la pantalla
        if self.player.sprite.rect.left <= 0:
            logging.info(f'Player {self.player_id} hit the left border.')
            # Aquí puedes añadir lógica adicional, como restar una vida o penalizar de otra forma
            self.lives -= 1
        elif self.player.sprite.rect.right >= self.screen_width:
            logging.info(f'Player {self.player_id} hit the right border.')
            # Aquí también puedes añadir lógica adicional, como restar una vida
            self.lives -= 1


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
        print('Enemigos restante: ', len(self.aliens))
        if(40>len(self.aliens) ):
            done = True
        
        # Devolver el nuevo estado, recompensa, indicador de fin
        return state, reward, done

    def calculate_reward(self, state, done, action):
        reward = 0

        if action == 3:
            reward += 5

        # Recompensa por destruir un enemigo (basado en incremento de puntaje)
        current_score = self.score
        if current_score > self.last_score:
            reward += (current_score - self.last_score)  # Ajusta el multiplicador según sea necesario
            logging.info(f'Player {self.player_id} scored points. Reward: {reward}')
            self.last_score = current_score

        if self.lives < self.last_lives:
            reward -= 300  # Ajusta la penalización si se pierde una vida
            logging.info(f'Player {self.player_id} lost a life. Penalty: -200')
            self.last_lives = self.lives

        if(40>len(self.aliens) ):
            reward += 1000

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

    def get_game_state(self):
        player_pos = self.player.sprite.rect.center

        # Alien más cercano
        closest_alien, alien_distance = self.get_closest(self.aliens, player_pos)

        # Láser de alien más cercano
        closest_alien_laser, laser_distance = self.get_closest(self.alien_lasers, player_pos)

        # Obstacle más cercano
        closest_obstacle, obstacle_distance = self.get_closest(self.blocks, player_pos)

        # Crear un diccionario con el estado
        game_state = [
            player_pos[0], player_pos[1],
            closest_alien.rect.center[0] if closest_alien else -1, closest_alien.rect.center[1] if closest_alien else -1, alien_distance,
            closest_alien_laser.rect.center[0] if closest_alien_laser else -1, closest_alien_laser.rect.center[1] if closest_alien_laser else -1, laser_distance,
            closest_obstacle.rect.center[0] if closest_obstacle else -1, #closest_obstacle.rect.center[1] if closest_obstacle else -1, obstacle_distance
        ]

        return game_state

class Game:
    def __init__(self, screen_width, screen_height, num_players, render=True):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.render = render
        self.num_players = num_players
        self.players = []

        # Crear una instancia de PlayerGame para cada jugador
        for i in range(num_players):
            self.players.append(PlayerGame(screen_width, screen_height, player_id=i))

        # Inicializar la pantalla solo si se requiere renderizado
        if self.render:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Space Invaders Multijugador')
        else:
            self.screen = None

    def run(self):
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

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.models = [self._build_model(input_size, hidden_size, output_size) for _ in range(population_size)]

    def _build_model(self, input_size, hidden_size, output_size):
        #model = tf.keras.models.Sequential([
        #    tf.keras.layers.InputLayer(input_shape=(input_size,)),
        ##    tf.keras.layers.Dense(hidden_size, activation='relu'),
        #   tf.keras.layers.Dense(output_size, activation='softmax')
        #])
        #model = tf.keras.models.Sequential([
        #    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
        #    tf.keras.layers.Dense(32, activation='relu'),
        #    tf.keras.layers.Dense(output_size, activation='softmax')
        #])
        model = Sequential([
            Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=9),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(4, activation='softmax')
        ])

        optimizer = Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        if(total_fitness == 0):
            total_fitness = 0.0001
        selection_probs = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.models, weights=selection_probs, k=2)
        return parents

    def crossoverOld(self, parent1, parent2, fitness1, fitness2):
        total_fitness = fitness1 + fitness2
        if(total_fitness != 0):
            weight1 = fitness1 / total_fitness
            weight2 = fitness2 / total_fitness
        else:
            weight1 = 0.5
            weight2 = 0.5
        child_weights = []

        for w1, w2 in zip(parent1.get_weights(), parent2.get_weights()):
            child_weights.append(weight1 * w1 + weight2 * w2)

        child = self._build_model(parent1.input_shape[-1], parent1.layers[1].units, parent1.layers[-1].units)
        child.set_weights(child_weights)
        return child

    def crossover(self, parent1, parent2, fitness1, fitness2):
        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()
        child_weights = []

        for w1, w2 in zip(parent1_weights, parent2_weights):
            # Realizar el crossover ponderado de los pesos basándose en el puntaje de los padres
            if((fitness1 + fitness2)>0):
                alpha = fitness1 / (fitness1 + fitness2)
            else:
                alpha = 0.5
            child_weight = alpha * w1 + (1 - alpha) * w2
            child_weights.append(child_weight)

        # Crear el modelo hijo con los pesos combinados
        child = tf.keras.models.clone_model(parent1)
        child.set_weights(child_weights)

        return child


    def mutate(self, model):
        weights = model.get_weights()
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.normal(0, 0.1, size=weights[i].shape)
        model.set_weights(weights)

    def evolve(self, fitness_scores):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)
            fitness1, fitness2 = fitness_scores[self.models.index(parent1)], fitness_scores[self.models.index(parent2)]
            child = self.crossover(parent1, parent2, fitness1, fitness2)
            self.mutate(child)
            new_population.append(child)
        self.models = new_population[:self.population_size]
        logging.info(f'Generation evolved. Best fitness score: {max(fitness_scores)}')

    def get_action(self, model, state):
        # Ejemplo de normalización de datos
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        state = np.array(state).reshape(1, -1)
        action_probs = model.predict(state)[0]
        return np.argmax(action_probs)

def main():
    pygame.init()
    screen_width = 800
    screen_height = 600
    num_players = 4
    population_size = num_players
    mutation_rate = 0.1
    crossover_rate = 0.7
    input_size = 9  # Example input size
    hidden_size = 16
    output_size = 4  # Number of possible actions

    game = Game(screen_width, screen_height, num_players, render=True)
    genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, input_size, hidden_size, output_size)

    # Definir la función run_player antes de iniciar el bucle de generaciones
    def run_player(genetic_algorithm, player, player_id):
        state = player.get_game_state()
        total_reward = 0
        done = False

        while not done:
            game.run()
            action = genetic_algorithm.get_action(genetic_algorithm.models[player_id], state)
            print(action)
            state, reward, done = player.step(action)
            print(state)
            total_reward += reward
            if reward > 0:
                logging.info(f'Player {player_id} received reward. Total reward: {total_reward}')
        
        player.reset()
        return total_reward
    
    for generation in range(30):
        logging.info(f'Starting generation {generation + 1}')
        fitness_scores = []
    
        with ThreadPoolExecutor() as executor:
            # Ejecutar a todos los jugadores en paralelo
            futures = [executor.submit(run_player, genetic_algorithm, player, i) for i, player in enumerate(game.players)]
            
            # Recopilar los resultados a medida que se completan
            for future in as_completed(futures):
                fitness_scores.append(future.result())
    
        logging.info(f'Fitness scores for generation {generation + 1}: {fitness_scores}')
    
        # Evolucionar los modelos basados en las puntuaciones de fitness
        genetic_algorithm.evolve(fitness_scores)
        game.reset()
   

if __name__ == "__main__":
    main()
