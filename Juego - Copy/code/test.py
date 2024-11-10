import tensorflow as tf
from main import Game  # Asegúrate de importar la clase Game correctamente
import pygame
import sys

# Cargar el modelo entrenado

loaded_model = tf.keras.models.load_model('best_model_gen_47.keras')


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
    pygame.time.set_timer(ALIENLASER, 600)

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
        # 1. Posición del jugador (normalizada por el ancho de la ventana)
        player_pos = np.array(game_state['player_position'])
        # 2. Distancias relativas de los enemigos
        alien_distances = game_state['alien_distances']
        while len(alien_distances) < expected_count:  # Asegura que haya 'expected_count' valores
            alien_distances.append(0)  # Relleno con 0 si hay menos de 'expected_count' valores
        
        # 4. Distancias relativas de los láseres enemigos
        enemy_laser_distances = game_state['enemy_laser_distances']
        while len(enemy_laser_distances) < expected_count:  # Asegura que haya 'expected_count' valores
            enemy_laser_distances.append(0)  # Relleno con 0 si hay menos de 'expected_count' valores
        
        # 5. Dirección de los enemigos
        enemy_direction = game_state['enemy_direction']

        # Concatenar todas las entradas en un solo vector de entrada
        input_data = np.concatenate([
            [player_pos],  # Solo la posición x del jugador, ya que el y no es relevante para la entrada
            alien_distances,  # Distancias relativas de los enemigos
            enemy_laser_distances,  # Distancias relativas de los láseres enemigos
            [enemy_direction]  # Dirección de los enemigos
        ])
        
        return input_data

play_game(loaded_model, game)