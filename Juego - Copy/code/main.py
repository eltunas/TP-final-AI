import pygame
import os
import logging
from game import Game
from AlgoritmoGenetico import GeneticAlgorithm
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


def main():
    pygame.init()  # Inicializar Pygame una sola vez al principio
    screen_width = 800
    screen_height = 600
    num_games = 20  # Número de juegos (1 jugador por juego)
    num_gens = 60
    mutation_rate = 0.1
    crossover_rate = 0.7
    input_size = 25
    hidden_size = 16
    output_size = 6
    num_threads = 4  # Número de threads para el executor
    runSync = True

    crossover_method='ponderados'
    elitism_rate=0.1

    genetic_algorithm = GeneticAlgorithm(num_games, mutation_rate, crossover_rate, input_size, hidden_size, output_size, crossover_method, elitism_rate)

    # Función para ejecutar cada juego individual con un jugador
    def run_single_game(genetic_algorithm, player_id, generation):
        # Crear el juego con un solo jugador
        game = Game(screen_width, screen_height, num_players=1, gen=generation, render=True)
        player = game.players[0]  # Acceder al único jugador en el juego

        state = player.get_game_state()
        total_reward = 0
        done = False

        while not done:
            game.run(background=True)  # Ejecuta el ciclo del juego
            action = genetic_algorithm.get_action(genetic_algorithm.models[player_id], state)
            state, reward, done = player.step(action)
            total_reward += reward

        player.reset()
        game.reset()  # Limpiar los recursos del juego después de cada ejecución
        return total_reward

    for generation in range(num_gens):
        logging.info(f'Starting generation {generation + 1}')
        fitness_scores = []

        if runSync:
            # Ejecución sincrónica: Ejecutar cada juego de forma secuencial
            for i in range(num_games):
                fitness_scores.append(run_single_game(genetic_algorithm, i, generation))
        else:
            # Ejecución asincrónica: Usar ThreadPoolExecutor para paralelizar los juegos
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(run_single_game, genetic_algorithm, i, generation) for i in range(num_games)]
                for future in as_completed(futures):
                    fitness_scores.append(future.result())

        logging.info(f'Fitness scores for generation {generation + 1}: {fitness_scores}')
        genetic_algorithm.evolve(fitness_scores)

    # Finalizar Pygame una sola vez al final
    pygame.quit()

   
os.environ["SDL_VIDEODRIVER"] = "dummy"
if __name__ == "__main__":
    main()

