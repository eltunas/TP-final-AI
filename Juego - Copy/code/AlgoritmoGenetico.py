import numpy as np
import tensorflow as tf
import pygame
import sys
import random
from multiprocessing import Pool
from main import Game
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations, input_size, output_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.input_size = input_size
        self.output_size = output_size
        self.population = self.initialize_population()
        self.level = 2
        self.win_count = 0

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.input_size,)),
            tf.keras.layers.Normalization(axis=-1),
            # Cambiar las capas ocultas a 100 neuronas
            tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
            # Capa de salida con softmax para las probabilidades de las acciones
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), 
                      loss='mean_squared_error')  # Usamos MSE para Deep Q-Learning
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
            time_limit = 30

            if generation > 50:
                time_limit = 180
            elif generation > 30:
                time_limit = 120
            elif generation>15:
                time_limit=60



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

            epsilon = max(0.01, 1 - generation * 0.01) # Decaimiento de exploración
            
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

        print(total_score)
        return total_score

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

    def crossover(self, parent1, parent2):
        """Realiza cruce entre dos individuos utilizando punto de corte para mantener diversidad genética."""
        child_model = self.build_model()  # Crea un nuevo modelo vacío
        
        for layer1, layer2, child_layer in zip(parent1.layers, parent2.layers, child_model.layers):
            # Asegúrate de que las capas sean Dense (que tienen pesos)
            if isinstance(layer1, tf.keras.layers.Dense) and isinstance(layer2, tf.keras.layers.Dense):
                weights1, biases1 = layer1.get_weights()
                weights2, biases2 = layer2.get_weights()

                mask = np.random.rand(*weights1.shape) > 0.5
                new_weights = np.where(mask, weights1, weights2)
                new_biases = np.where(mask[0], biases1, biases2)  # Máscara para sesgo también

                child_layer.set_weights([new_weights, new_biases])
            # Si es una capa que no tiene pesos, simplemente la copiamos tal cual
            else:
                child_layer.set_weights(layer1.get_weights())

        return child_model

    def mutate(self, model, generation):
        """Mutación aleatoria con mayor probabilidad de cambio en generaciones tempranas."""
        for layer in model.layers:
            # Para capas como Dense, que tienen pesos y sesgos
            if isinstance(layer, tf.keras.layers.Dense):
                weights, biases = layer.get_weights()

                # Aumentamos la perturbación en las primeras generaciones
                mutation_scale = max(0.1, 1.0 / (generation + 1))  # Reduce la perturbación conforme aumentan las generaciones

                # Si la mutación está habilitada, realizamos el cambio con la escala adaptativa
                if np.random.rand() < self.mutation_rate:
                    weights += np.random.normal(scale=mutation_scale, size=weights.shape)  # Perturbación para pesos
                if np.random.rand() < self.mutation_rate:
                    biases += np.random.normal(scale=mutation_scale, size=biases.shape)  # Perturbación para los sesgos

                # Establecer los nuevos pesos y sesgos
                layer.set_weights([weights, biases])

            # Para capas de Normalización (como tf.keras.layers.Normalization)
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                # Las capas de normalización pueden tener más de dos valores, por ejemplo gamma y beta
                weights = layer.get_weights()

                # Verificar si se obtuvo gamma y beta (típicamente, se deben manejar como gamma y beta)
                if len(weights) == 2:
                    gamma, beta = weights
                    # Aumentamos la perturbación en las primeras generaciones
                    mutation_scale = max(0.1, 1.0 / (generation + 1))  # Reduce la perturbación conforme aumentan las generaciones

                    # Si la mutación está habilitada, realizamos el cambio con la escala adaptativa
                    if np.random.rand() < self.mutation_rate:
                        gamma += np.random.normal(scale=mutation_scale, size=gamma.shape)  # Perturbación para gamma
                    if np.random.rand() < self.mutation_rate:
                        beta += np.random.normal(scale=mutation_scale, size=beta.shape)  # Perturbación para beta

                    # Establecer los nuevos gamma y beta
                    layer.set_weights([gamma, beta])
                else:
                    # Si la capa de normalización tiene más pesos, manejarlos de acuerdo a su estructura
                    print(f"Advertencia: La capa de Normalization tiene {len(weights)} pesos, no se mutan.")
                    
            # Otros tipos de capas pueden ser manejados aquí si es necesario (por ejemplo, Dropout, BatchNormalization, etc.)
        return model

    def create_new_generation(self, best_individuals, generation):
        """Crea una nueva generación usando los mejores individuos y aplicando cruce y mutación."""
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(best_individuals, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, generation)
            new_population.append(child)
        return new_population

    def evolve(self):
        """Ejecuta el ciclo de evolución a través de varias generaciones."""
        for generation in range(self.num_generations):
            print(f'Inciando generacion {generation}')

            game_instances = [Game(800, 600) for _ in range(self.population_size)]

            fitness_scores = []
    
                # Crea un ThreadPoolExecutor con tantos hilos como tamaño de la población
            with ThreadPoolExecutor(max_workers=self.population_size) as executor:
                # Define una lista de tareas para cada modelo en la población, asociada a su instancia de juego
                futures = [
                    executor.submit(self.evaluate_fitness, model, generation, game_instances[i], i)
                    for i, model in enumerate(self.population)
                ]
                
                # Recolecta los resultados a medida que se completan las tareas
                for future in as_completed(futures):
                    fitness_scores.append(future.result())

            print(f'Scores de la generación {generation}: {fitness_scores}')
            # Seleccionar los mejores individuos y generar la nueva población
            best_individuals = self.select_best_individuals(fitness_scores)

            self.population = self.create_new_generation(best_individuals, generation)
            print(f"Generación {generation}: Mejor puntaje = {max(fitness_scores)}")

        # Obtener el mejor modelo entrenado
        # Al final del ciclo evolutivo, guarda el mejor modelo
        best_model = best_individuals[0]
        
        return best_model  # Devuelve el mejor modelo entrenado


