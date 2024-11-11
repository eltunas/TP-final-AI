import numpy as np
import tensorflow as tf
import pygame
import sys
import random
from multiprocessing import Pool
from main import Game
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations, input_size, output_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.input_size = input_size
        self.output_size = output_size
        self.loadModel = False
        self.modelLoaded = False
        self.modelToLoadPath = './best_model_weights_gen_205.weights.h5'
        self.population = self.initialize_population()
        self.level = 2
        self.win_count = 0
        self.generation = 0 


        # Listas para almacenar el progreso de cada generación
        self.best_scores = []
        self.best_total_scores = []
        self.best_epsilons = []

    def build_model(self):
        model = Sequential([
            Dense(128, activation='swish', kernel_initializer='he_uniform', input_dim=self.input_size),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='leaky_relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='swish', kernel_regularizer=l2(0.01)),
            Dense(self.output_size, activation='softmax')
        ])


        optimizer = Adam(learning_rate=0.001)  

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if self.loadModel:
            model = tf.keras.models.load_model('best_model_gen_200.keras')

        return model


    def initialize_population(self):
        """Inicializa una población de redes con pesos aleatorios."""
        population = []
        for _ in range(self.population_size):
            model = self.build_model()  # Crea el modelo con pesos aleatorios
            population.append(model)    # Guarda el modelo completo
        return population

    def evaluate_fitness(self,  model, generation,  game, instance_id):
        """Evalúa la aptitud de una red neuronal ejecutándola en el juego."""

        print(f'Agente {instance_id} Iniciado. Level: {self.level}')
        
        total_score = 0
        game.reset()  # Reinicia el juego
        clock = pygame.time.Clock()
        done = False

        if self.level == 1:
            shoot = False
            shoot_speed = 0
            time_limit= 15

            if generation > 20:
                time_limit = 60

            elif generation > 10:
                time_limit = 30
        
        elif self.level == 2:
            shoot = True
            shoot_speed = 600
            time_limit = 120

            #if generation > 50:
                #time_limit = 180
            #elif generation > 30:
                #time_limit = 120
            #elif generation>15:
                #time_limit=60



        elif self.level == 3:
            shoot = True
            shoot_speed = 300
            time_limit = 180

        start_time = time.time()  # Iniciar el temporizador

        pygame.font.init()
        font = pygame.font.Font(None, 36)  # Fuente predeterminada, tamaño 36

        ALIENLASER = pygame.USEREVENT + 1
        pygame.time.set_timer(ALIENLASER, shoot_speed)

        while not done:
            elapsed_time = int(time.time() - start_time)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == ALIENLASER and shoot == True:
                    game.alien_shoot()

            game_state = game.get_game_state()
            input_data = self.preprocess_game_state(game_state)
            input_data = np.nan_to_num(input_data, nan=0, posinf=1e6, neginf=-1e6)
            input_data = np.expand_dims(input_data, axis=0)

            epsilon = max(0.01, 1 * (0.95 ** generation)) # Decaimiento de exploración
            
            action_probs = model.predict(input_data, verbose=0)[0]

            action = np.random.choice(len(action_probs)) if np.random.rand() < epsilon else np.argmax(action_probs)

            _, reward, done = game.step(action, start_time)
            total_score += reward

            game.screen.fill((30, 30, 30))
            game.run()
            pygame.display.flip()
            clock.tick(60)

            if elapsed_time > time_limit:
                print("Time Finished")
                done = True
                total_score -= 1000
            
            if not game.aliens.sprites():
                print("Ganaste")
                self.win_count += 1
                if self.win_count >= 5 and self.level != 3:
                    self.level += 1
                    self.win_count = 0
                done = True

        print(f"Puntaje{game.score}, Fitness: {total_score}")
        return total_score, game.score, epsilon

    def export_best_model(self, fitness_scores):
        # Get the index of the best model based on fitness scores
        best_index = np.argmax(fitness_scores)
        best_model = self.population[best_index]
        # Save the weights to a file
        best_model.save(f'best_model_gen_{self.generation}.keras')

        self.generation+=1

    def preprocess_game_state(self, game_state, expected_count=5):
        """Preprocesa el estado del juego para convertirlo en un vector de entrada listo para el modelo."""
        
        # 1. Posición del jugador (normalizada en el eje x)
        player_pos = np.array([game_state['player_position']])  # Convertido a array para la concatenación posterior
        
        # 2. Distancias relativas de los enemigos en el eje x
        alien_distances = game_state['alien_x_distances']
        if len(alien_distances) < expected_count:
            # Relleno con 0 hasta alcanzar el número esperado
            alien_distances += [0] * (expected_count - len(alien_distances))
        alien_distances = np.array(alien_distances)  # Convertido a array para el modelo
        
        # 3. Distancias relativas de los láseres enemigos en el eje x
        enemy_laser_distances = game_state['enemy_laser_x_distances']
        if len(enemy_laser_distances) < expected_count:
            # Relleno con 0 hasta alcanzar el número esperado
            enemy_laser_distances += [0] * (expected_count - len(enemy_laser_distances))
        enemy_laser_distances = np.array(enemy_laser_distances)  # Convertido a array para el modelo
        
        # 4. Dirección general de los enemigos en el eje x
        enemy_direction = np.array([game_state['enemy_direction']])  # Convertido a array para la concatenación

        # 5. Concatenar todas las entradas en un solo vector
        input_data = np.concatenate([
            player_pos,               # Posición normalizada del jugador en el eje x
            alien_distances,          # Distancias relativas de los enemigos más cercanos
            enemy_laser_distances,    # Distancias relativas de los láseres enemigos más cercanos
            enemy_direction           # Dirección general de los enemigos
        ])
        
        return input_data

    def select_best_individuals(self, fitness_scores, num_best=2):
        """Selecciona los mejores individuos basándose en sus puntajes de aptitud."""
        best_indices = np.argsort(fitness_scores)[-num_best:]
        return [self.population[i] for i in best_indices]

    def crossover(self, parent1, parent2, fitness_scores):
        """Realiza un cruce ponderado en función de la aptitud entre dos padres para generar un hijo."""
        fitness1 = np.argsort(fitness_scores)[-1]
        fitness2 = np.argsort(fitness_scores)[-2]
        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()
        child_weights = []

        # Usar crossover ponderado según la aptitud
        for w1, w2 in zip(parent1_weights, parent2_weights):
            # Calcular alpha ponderado
            alpha = fitness1 / (fitness1 + fitness2) if (fitness1 + fitness2) > 0 else 0.5
            # Realizar el cruce ponderado de los pesos
            child_weight = alpha * w1 + (1 - alpha) * w2
            child_weights.append(child_weight)

        # Crear el modelo hijo con los pesos combinados
        child = tf.keras.models.clone_model(parent1)
        child.set_weights(child_weights)

        return child

    def mutate(self, model):
        """Realiza una mutación en el modelo cambiando los pesos aleatoriamente."""
        weights = model.get_weights()
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                # Se aplica una pequeña mutación normal a los pesos
                weights[i] += np.random.normal(0, 0.1, size=weights[i].shape)
        model.set_weights(weights)

        return model

    def create_new_generation(self, best_individuals, fitness_scores):
        """Crea una nueva generación usando los mejores individuos y aplicando cruce y mutación."""
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(best_individuals, 2)
            child = self.crossover(parent1, parent2, fitness_scores)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def evolve(self):
        for generation in range(self.num_generations):
            print(f'Iniciando generacion {generation}')
            game_instances = [Game(800, 600) for _ in range(self.population_size)]
            fitness_scores = []
            scores = []
            epsilons = []

            with ThreadPoolExecutor(max_workers=self.population_size) as executor:
                futures = [
                    executor.submit(self.evaluate_fitness, model, generation, game_instances[i], i)
                    for i, model in enumerate(self.population)
                ]
                for future in as_completed(futures):
                    total_score, game_score, epsilon = future.result()
                    fitness_scores.append(total_score)
                    scores.append(game_score)
                    epsilons.append(epsilon)

            # Encuentra el índice del mejor puntaje en esta generación
            best_index = np.argmax(fitness_scores)
            self.best_scores.append(scores[best_index])
            self.best_total_scores.append(fitness_scores[best_index])
            self.best_epsilons.append(epsilons[best_index])

            print(f'Scores de la generación {generation}: {fitness_scores}')
            # Seleccionar los mejores individuos y generar la nueva población
            best_individuals = self.select_best_individuals(fitness_scores)

            self.population = self.create_new_generation(best_individuals, fitness_scores)

            self.export_best_model(fitness_scores)
            print(f"Generación {generation}: Mejor puntaje = {max(fitness_scores)}")

        # Obtener el mejor modelo entrenado
        # Al final del ciclo evolutivo, guarda el mejor modelo
        best_model = best_individuals[0]
        
        return best_model  # Devuelve el mejor modelo entrenado


