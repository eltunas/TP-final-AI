import numpy as np
import tensorflow as tf
import pygame
import sys
import random
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations, input_size, output_size, game):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.input_size = input_size
        self.output_size = output_size
        self.game = game  # La instancia del juego para evaluar cada red
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
            model = self.build_model()
            weights = [layer.get_weights() for layer in model.layers]
            population.append(weights)
        return population

    def evaluate_fitness(self, weights, instance_id, offset_step=30):
        """Evalúa la aptitud de una red neuronal ejecutándola en el juego."""
        # Construir y configurar el modelo con los pesos dados
        model = self.build_model()
        for layer, weight in zip(model.layers, weights):
            layer.set_weights(weight)

        # Configuración del temporizador para disparos de aliens
        ALIENLASER = pygame.USEREVENT + 1
        pygame.time.set_timer(ALIENLASER, 100)

        total_score = 0
        self.game.reset()  # Reiniciar el juego
        clock = pygame.time.Clock()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == ALIENLASER:
                    self.game.alien_shoot()

            # Obtener el estado actual del juego y predecir la acción
            game_state = self.game.get_game_state()
            input_data = self.preprocess_game_state(game_state).reshape(1, -1)
            action = np.argmax(model.predict(input_data))
            _, reward, done = self.game.step(action)
            total_score += reward

            # Actualizar la pantalla y rellenar el fondo
            self.game.screen.fill((30, 30, 30))  # Rellena el fondo con color gris oscuro
            self.game.run()  # Ejecuta la lógica de actualización de la pantalla
            pygame.display.flip()  # Actualiza la ventana
            clock.tick(120)  # Controla la velocidad del bucle

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

    def select_best_individuals(self, fitness_scores, num_best=5):
        """Selecciona los mejores individuos basándose en sus puntajes de aptitud."""
        best_indices = np.argsort(fitness_scores)[-num_best:]
        return [self.population[i] for i in best_indices]

    def crossover(self, parent1, parent2):
        """Realiza cruce entre dos individuos para producir un nuevo conjunto de pesos."""
        child = []
        for p1_layer, p2_layer in zip(parent1, parent2):
            new_weights = [(p1 + p2) / 2 for p1, p2 in zip(p1_layer, p2_layer)]
            child.append(new_weights)
        return child

    def mutate(self, weights):
        """Aplica mutación aleatoria a un conjunto de pesos."""
        for layer in weights:
            for i in range(len(layer)):
                if np.random.rand() < self.mutation_rate:
                    layer[i] += np.random.normal()
        return weights

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
            # Ejecutar evaluación en paralelo para todos los individuos de la generación
            with ThreadPoolExecutor(max_workers=self.population_size) as executor:
                # Asigna una posición escalonada para cada individuo
                args = [(self.population[i], i) for i in range(self.population_size)]
                fitness_scores = list(executor.map(lambda arg: self.evaluate_fitness(*arg), args))

            # Crear la siguiente generación usando los puntajes de aptitud
            self.create_new_generation(fitness_scores)

            # Imprimir el mejor puntaje de la generación actual
            print(f"Generación {generation}: Mejor puntaje = {max(fitness_scores)}")

        # Devuelve los pesos del mejor modelo entrenado
        best_weights = self.select_best_individuals(fitness_scores, num_best=1)[0]
        final_model = self.build_model()
        for layer, weight in zip(final_model.layers, best_weights):
            layer.set_weights(weight)
        
        return final_model  # Devuelve el mejor modelo entrenado

