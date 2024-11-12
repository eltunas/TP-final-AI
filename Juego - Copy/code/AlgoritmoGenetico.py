import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import random
import logging
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, input_size, hidden_size, output_size, crossover_method='uniform', elitism_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.loadModel = False
        self.modelLoaded = False
        self.crossover_method = crossover_method 
        self.elitism_rate = elitism_rate 
        self.modelToLoadPath = './best_model_weights_gen_112.weights.h5'
        self.models = [self._build_model(input_size, hidden_size, output_size) for _ in range(population_size)]
        self.generation = 0
     

    def _build_model(self, input_size, hidden_size, output_size):

        model = Sequential([
            Input(shape=(input_size,)),  # Explicitly define the input shape
            Dense(128, activation='swish', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='leaky_relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='swish', kernel_regularizer=l2(0.01)),
            Dense(output_size, activation='softmax')
        ])

        if(self.loadModel and not self.modelLoaded):
            model.load_weights(self.modelToLoadPath)
            self.modelLoaded = True  

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    # Nota volvi al metodo de probabilidad por que funciona mejor pero con mejoras
    def select_parents(self, fitness_scores):
        # Asegurarnos de que todos los puntajes sean positivos sumando el valor absoluto del puntaje mínimo
        min_score = min(fitness_scores)
        adjusted_scores = [score - min_score for score in fitness_scores]  # Ajuste para que el mínimo sea 0

        # Calcular las probabilidades en base a los puntajes ajustados
        total_score = sum(adjusted_scores)
        
        if total_score == 0:
            # Si todos los puntajes ajustados son 0, seleccionar padres al azar
            parent1, parent2 = random.sample(self.models, 2)
        else:
            # Si hay un puntaje positivo total, seleccionar en base a probabilidades
            probabilities = [score / total_score for score in adjusted_scores]
            
            # Seleccionar dos padres utilizando la distribución de probabilidad en base a self.models
            parent1 = random.choices(self.models, weights=probabilities, k=1)[0]
            parent2 = random.choices(self.models, weights=probabilities, k=1)[0]
            
            # Asegurarse de que los dos padres sean diferentes
            while parent2 == parent1:
                parent2 = random.choices(self.models, weights=probabilities, k=1)[0]

        return parent1, parent2

    def crossover(self, parent1, parent2, fitness1, fitness2):
        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()
        child_weights = []

        #Uso ponderado el 80% de la veces, dejo uniforme el 20% restante para agregar algo de variacion
        self.crossover_method = random.choices(['ponderado', 'uniform'], weights=[0.8, 0.2], k=1)[0]

        if self.crossover_method == 'uniform':
            # Crossover uniforme
            for w1, w2 in zip(parent1_weights, parent2_weights):
                mask = np.random.randint(0, 2, size=w1.shape).astype(bool)
                child_weight = np.where(mask, w1, w2)
                child_weights.append(child_weight)
        elif self.crossover_method == 'point':
            # Crossover por puntos
            crossover_point = random.randint(1, len(parent1_weights) - 1)
            child_weights = parent1_weights[:crossover_point] + parent2_weights[crossover_point:]
        else:
            # Crossover ponderado (por defecto)
            alpha = fitness1 / (fitness1 + fitness2) if (fitness1 + fitness2) > 0 else 0.5
            for w1, w2 in zip(parent1_weights, parent2_weights):
                child_weight = alpha * w1 + (1 - alpha) * w2
                child_weights.append(child_weight)

        # Crear el modelo hijo con los pesos combinados
        child = tf.keras.models.clone_model(parent1)
        child.set_weights(child_weights)

        return child

    def mutate(self, model, decay_factor = 0.99):
        weights = model.get_weights()
        factor = 0.1 * (decay_factor ** self.generation)
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.normal(0, factor, size=weights[i].shape)
        model.set_weights(weights)

    def evolve(self, fitness_scores):
        
        # Ordenar modelos y puntajes de fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Orden descendente
        sorted_models = [self.models[i] for i in sorted_indices]
        #sorted_fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Elitismo: conservar los mejores individuos
        elite_count = 2
        new_population = sorted_models[:elite_count].copy()


        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)
            fitness1, fitness2 = fitness_scores[self.models.index(parent1)], fitness_scores[self.models.index(parent2)]
            child = self.crossover(parent1, parent2, fitness1, fitness2)
            self.mutate(child)
            new_population.append(child)
        self.models = new_population[:self.population_size]
        average_fitness_score = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        logging.info(f'Generation evolved. Best fitness score: {max(fitness_scores)}, Average fitness score: {average_fitness_score}')
        self.generation += 1
        self.export_best_model_weights(fitness_scores)

    def export_best_model_weights(self, fitness_scores):
        # Get the index of the best model based on fitness scores
        best_index = np.argmax(fitness_scores)
        best_model = self.models[best_index]
        # Save the weights to a file
        best_model.save_weights(f'best_model_weights_gen_{self.generation}.weights.h5')


    def get_action(self, model, state, epsilon=0.01):
        # Exploración vs. Explotación
        if np.random.rand() < (epsilon / (self.generation + 1)):
            # Exploración: elige una acción aleatoria
            return np.random.randint(0, model.output_shape[-1])
        else:
            # Explotación: elige la mejor acción predicha por el modelo
            state = (state - np.mean(state)) / (np.std(state) + 1e-8)
            state = np.array(state).reshape(1, -1)
            action_probs = model.predict(state, verbose=0)[0]
            return np.argmax(action_probs)