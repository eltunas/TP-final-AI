import tensorflow as tf
from main import Game  # Asegúrate de importar la clase Game correctamente
import pygame
import sys

# Cargar el modelo entrenado

loaded_model = tf.keras.models.load_model('best_model_gen_67.keras')


# Crear una instancia del juego
pygame.init()  # Asegúrate de inicializar Pygame
pygame.font.init()
game = Game(800, 600)

import numpy as np
import pygame
import time

def play_game(model, game):
    game.reset()  # Reiniciar el juego
    clock = pygame.time.Clock()
    done = False
    epsilon = 0.1  # Probabilidad de exploración (si quieres agregar exploración aleatoria)

    total_score=0


    start_time = time.time()  # Para limitar el tiempo de juego si es necesario
    time_limit = 180  # Puedes ajustar este valor según el tiempo que desees que dure la simulación

    ALIENLASER = pygame.USEREVENT + 1
    pygame.time.set_timer(ALIENLASER, 800)

    while not done:
        # Calcular el tiempo transcurrido
        elapsed_time = int(time.time() - start_time)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == ALIENLASER:
                    game.alien_shoot()

        # Obtener el estado del juego
        game_state = game.get_game_state()  # Asegúrate de que este método existe y devuelve el estado correcto
        input_data = preprocess_game_state(game_state)
        input_data = np.nan_to_num(input_data, nan=0, posinf=1e6, neginf=-1e6)
        input_data = np.expand_dims(input_data, axis=0)

        # Obtener las probabilidades de las acciones
        action_probs = model.predict(input_data, verbose=0)[0]

        # Decidir si explorar o explotar
        if np.random.rand() < epsilon:
            action = np.random.choice(len(action_probs))  # Exploración (acción aleatoria)
        else:
            action = np.argmax(action_probs)  # Explotación (acción con mayor probabilidad)
        print(action)

        # Ejecutar la acción en el juego
        _, reward, done = game.step(action, start_time)
        total_score += reward

        game.screen.fill((30, 30, 30))
        game.run()
        pygame.display.flip()
        clock.tick(60)



    print(f"Juego terminado. Su puntaje fue: {total_score}")



def preprocess_game_state(game_state, expected_count=5):
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

play_game(loaded_model, game)