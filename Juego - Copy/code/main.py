# main.py
import pygame
import sys
from Game import Game
from AlgoritmoGenetico import GeneticAlgorithm
import numpy as np

# Parámetros del algoritmo genético
population_size = 20
mutation_rate = 0.1
num_generations = 20
input_size = 11  # Tamaño del vector de estado
output_size = 4  # Cuatro acciones posibles: nada, izquierda, derecha, disparar

def main():
    screen_width = 800
    screen_height = 600
    GameClass = lambda renderizar: Game(screen_width, screen_height, renderizar)

    # Crear una instancia del algoritmo genético
    gen_algo = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=mutation_rate,
        num_generations=num_generations,
        input_size=input_size,
        output_size=output_size,
        game_class=GameClass,
        screen_width=screen_width,
        screen_height=screen_height
    )

    # Ejecutar el proceso evolutivo para entrenar la red
    print("Entrenando la red neuronal mediante algoritmo genético...")
    trained_model = gen_algo.evolve()  # Este será el mejor modelo entrenado

    # Usar el modelo entrenado para jugar el juego automáticamente
    print("Usando el mejor modelo para jugar el juego...")
    game = Game(screen_width, screen_height, renderizar=True)  # Renderizar para visualizar
    clock = pygame.time.Clock()
    done = False

    ALIENLASER = pygame.USEREVENT + 1
    pygame.time.set_timer(ALIENLASER, 600)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == ALIENLASER:
                game.alien_shoot()

        # Obtener el estado del juego y procesarlo para el modelo
        game_state = game.get_game_state()
        input_data = gen_algo.preprocess_game_state(game_state).reshape(1, -1)

        # Obtener la acción basada en el modelo entrenado
        action = np.argmax(trained_model.predict(input_data, verbose=0))


        # Ejecutar la acción en el juego
        _, _, done = game.step(action)

        # Controlar la velocidad del bucle
        clock.tick(60)

    game.reset()  # Reiniciar el juego para la ejecución automática

    # Finalizar Pygame
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
