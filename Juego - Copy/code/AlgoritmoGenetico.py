from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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
        model = Sequential([
            Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=11),
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