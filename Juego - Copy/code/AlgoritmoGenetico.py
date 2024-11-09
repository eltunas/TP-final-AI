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

    def build_model(self):
        """Construye un modelo de red neuronal."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])
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

        print(f'Agente {instance_id} Iniciado')
        
        total_score = 0
        game.reset()  # Reinicia el juego
        clock = pygame.time.Clock()
        done = False

        shoot = False
        if generation > 5:
            shoot = True
        shoot_speed= 800
        if generation > 10:
            shoot_speed = 500

        start_time = time.time()  # Iniciar el temporizador

        if generation > 10:
            time_limit = 180
        elif generation > 5:
            time_limit = 120
        elif generation > 2:
            time_limit = 60
        else:
            time_limit = 30
        

        pygame.font.init()
        font = pygame.font.Font(None, 36)  # Fuente predeterminada, tamaño 36

        ALIENLASER = pygame.USEREVENT + 1
        pygame.time.set_timer(ALIENLASER, 600)

        while not done:
            elapsed_time = int(time.time() - start_time)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == ALIENLASER and shoot == True:
                    game.alien_shoot()

            game_state = game.get_game_state()
            input_data = self.preprocess_game_state(game_state).reshape(1, -1)

            epsilon = 0.1  # Probabilidad de explorar aleatoriamente

            # Obtener las probabilidades de las acciones
            action_probs = model.predict(input_data, verbose=0)[0]

            # Decidir si explorar o explotar
            if np.random.rand() < epsilon:
                # Exploración: elegir una acción aleatoria
                action = np.random.choice(len(action_probs))
            else:
                # Explotación: elegir la acción con la mayor probabilidad
                action = np.argmax(action_probs) 

            _, reward, done = game.step(action, start_time)
            total_score += reward

            game.screen.fill((30, 30, 30))
            game.run()

            timer_text = font.render(f"Tiempo: {elapsed_time}s", True, (255, 255, 255))
            text_rect = timer_text.get_rect(center=(game.screen.get_width() // 2, 20))
            game.screen.blit(timer_text, text_rect)

            pygame.display.flip()
            clock.tick(120)

            if elapsed_time > time_limit:
                print("Time Finished")
                done = True
                total_score -= 1000

        print(total_score)
        return total_score

    def preprocess_game_state(self, game_state):
        """Preprocesa el estado del juego para convertirlo en una entrada válida para el modelo."""
        player_pos = np.array(game_state['player_position'])
        alien_pos, alien_distance = game_state['closest_alien']
        laser_pos, laser_distance = game_state['closest_alien_laser']
        obstacle_pos, obstacle_distance = game_state['closest_obstacle']
        
        alien_pos = np.array(alien_pos) if alien_pos else np.zeros(2)
        laser_pos = np.array(laser_pos) if laser_pos else np.zeros(2)
        obstacle_pos = np.array(obstacle_pos) if obstacle_pos else np.zeros(2)
        
        input_data = np.concatenate([player_pos, alien_pos, [alien_distance], laser_pos, [laser_distance], obstacle_pos, [obstacle_distance]])
        return input_data

    def select_best_individuals(self, fitness_scores, num_best=2):
        """Selecciona los mejores individuos basándose en sus puntajes de aptitud."""
        best_indices = np.argsort(fitness_scores)[-num_best:]
        return [self.population[i] for i in best_indices]

    def crossover(self, parent1, parent2):
        """Realiza cruce entre dos individuos para producir un nuevo conjunto de pesos."""
        child_model = self.build_model()  # Crear un nuevo modelo vacío
        for layer1, layer2, child_layer in zip(parent1.layers, parent2.layers, child_model.layers):
            # Promediar los pesos entre los padres
            weights1, biases1 = layer1.get_weights()
            weights2, biases2 = layer2.get_weights()

            new_weights = (weights1 + weights2) / 2
            new_biases = (biases1 + biases2) / 2
            child_layer.set_weights([new_weights, new_biases])

        return child_model

    def mutate(self, model):
        """Aplica mutación aleatoria a un modelo."""
        for layer in model.layers:
            weights, biases = layer.get_weights()

            # Mutación en los pesos
            if np.random.rand() < self.mutation_rate:
                weights += np.random.normal(scale=0.1, size=weights.shape)
            if np.random.rand() < self.mutation_rate:
                biases += np.random.normal(scale=0.1, size=biases.shape)

            layer.set_weights([weights, biases])

        return model

    def create_new_generation(self, best_individuals):
        """Crea una nueva generación usando los mejores individuos y aplicando cruce y mutación."""
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(best_individuals, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
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

            self.population = self.create_new_generation(best_individuals)
            print(f"Generación {generation}: Mejor puntaje = {max(fitness_scores)}")

        # Obtener el mejor modelo entrenado
        # Al final del ciclo evolutivo, guarda el mejor modelo
        best_model = best_individuals[0]
        
        return best_model  # Devuelve el mejor modelo entrenado


