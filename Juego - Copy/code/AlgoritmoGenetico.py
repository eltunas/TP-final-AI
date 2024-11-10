# AlgoritmoGenetico.py
import numpy as np
import tensorflow as tf
import random
import pygame
from joblib import Parallel, delayed


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations, input_size, output_size, game_class, screen_width, screen_height):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.input_size = input_size
        self.output_size = output_size
        self.game_class = game_class  # Clase del juego para crear instancias
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.population = self.initialize_population()

    def build_model(self):
        """Construye un modelo de red neuronal."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])
        return model

    def initialize_population(self):
        """Inicializa una población de redes con pesos aleatorios."""
        population = []
        for _ in range(self.population_size):
            model = self.build_model()
            weights = model.get_weights()
            population.append(weights)
        return population

    def evaluate_fitness(self, weights):
        """Evalúa la aptitud de una red neuronal ejecutándola en el juego."""
        model = self.build_model()
        model.set_weights(weights)

        total_reward = 0
        game_instance = self.game_class(renderizar=False)  # Ejecutar sin renderizado gráfico
        game_instance.reset()  # Reinicia el juego
        done = False

        ALIENLASER = pygame.USEREVENT + 1
        pygame.time.set_timer(ALIENLASER, 100)

        max_steps = 1000  # Establece un límite de pasos por episodio
        steps = 0

        while not done and steps < max_steps:
            # No es necesario manejar eventos si no hay renderizado

            game_state = game_instance.get_game_state()
            input_data = self.preprocess_game_state(game_state).reshape(1, -1)
            action = np.argmax(model.predict(input_data, verbose=0))
            _, reward, done = game_instance.step(action)
            total_reward += reward
            steps += 1

        pygame.time.set_timer(ALIENLASER, 0)  # Detener el evento

        # Penalizar si se alcanzó el máximo de pasos sin terminar el juego
        if steps >= max_steps and game_instance.lives > 0:
            total_reward -= 50  # Penalización por no completar el juego

        return total_reward

    def preprocess_game_state(self, game_state):
        """Preprocesa el estado del juego para convertirlo en una entrada válida para el modelo."""
        # Normalizar posiciones y distancias
        player_pos = np.array(game_state['player_position']) / [self.screen_width, self.screen_height]
        alien_pos, alien_distance = game_state['closest_alien']
        if alien_pos:
            alien_pos = np.array(alien_pos) / [self.screen_width, self.screen_height]
            alien_distance /= np.sqrt(self.screen_width**2 + self.screen_height**2)
        else:
            alien_pos = np.zeros(2)
            alien_distance = 1.0

        laser_pos, laser_distance = game_state['closest_alien_laser']
        if laser_pos:
            laser_pos = np.array(laser_pos) / [self.screen_width, self.screen_height]
            laser_distance /= np.sqrt(self.screen_width**2 + self.screen_height**2)
        else:
            laser_pos = np.zeros(2)
            laser_distance = 1.0

        obstacle_pos, obstacle_distance = game_state['closest_obstacle']
        if obstacle_pos:
            obstacle_pos = np.array(obstacle_pos) / [self.screen_width, self.screen_height]
            obstacle_distance /= np.sqrt(self.screen_width**2 + self.screen_height**2)
        else:
            obstacle_pos = np.zeros(2)
            obstacle_distance = 1.0

        input_data = np.concatenate([
            player_pos,
            alien_pos, [alien_distance],
            laser_pos, [laser_distance],
            obstacle_pos, [obstacle_distance]
        ])

        return input_data

    def select_parents(self, fitness_scores):
        """Implementa la selección por torneo."""
        parents = []
        for _ in range(self.population_size):
            participants = random.sample(list(zip(self.population, fitness_scores)), k=3)
            winner = max(participants, key=lambda x: x[1])
            parents.append(winner[0])
        return parents

    def crossover(self, parent1, parent2):
        """Realiza cruce uniforme entre dos individuos."""
        child_weights = []
        for w1, w2 in zip(parent1, parent2):
            mask = np.random.randint(0, 2, size=w1.shape)
            child_w = np.where(mask, w1, w2)
            child_weights.append(child_w)
        return child_weights

    def mutate(self, weights):
        """Aplica mutación aleatoria a un conjunto de pesos."""
        for i in range(len(weights)):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.normal(0, 0.1, size=weights[i].shape)
                weights[i] += mutation
        return weights

    def create_new_generation(self, parents, fitness_scores):
        """Crea una nueva generación usando los padres seleccionados."""
        new_population = []
        num_elite = max(1, self.population_size // 10)  # Elitismo: conservar el top 10%
        elite_indices = np.argsort(fitness_scores)[-num_elite:]
        elite = [self.population[i] for i in elite_indices]
        new_population.extend(elite)

        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def evolve(self):
        """Ejecuta el ciclo de evolución a través de varias generaciones."""
        for generation in range(self.num_generations):
            print(f"\nGeneración {generation + 1}")

            # Evaluar la aptitud de cada individuo en paralelo
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(self.evaluate_fitness)(weights) for weights in self.population
            )

            # Imprimir los puntajes
            for idx, score in enumerate(fitness_scores):
                print(f"Individuo {idx + 1}: Puntuación = {score}")

            # Seleccionar padres
            parents = self.select_parents(fitness_scores)

            # Crear una nueva generación
            self.population = self.create_new_generation(parents, fitness_scores)

            # Imprimir el mejor puntaje de la generación actual
            best_score = max(fitness_scores)
            avg_score = sum(fitness_scores) / len(fitness_scores)
            print(f"Mejor puntuación de la generación {generation + 1}: {best_score}")
            print(f"Puntuación promedio de la generación {generation + 1}: {avg_score}")

        # Al final del ciclo evolutivo, guarda el mejor modelo
        best_index = np.argmax(fitness_scores)
        best_weights = self.population[best_index]
        final_model = self.build_model()
        final_model.set_weights(best_weights)

        return final_model  # Devuelve el mejor modelo entrenado