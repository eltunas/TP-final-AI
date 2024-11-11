import tensorflow as tf
import pygame
import numpy as np
import sys
import time
import imageio

from main import Game  # Asegúrate de importar la clase Game correctamente

import tensorflow as tf
from main import Game  # Asegúrate de importar la clase Game correctamente
import pygame
import sys
import numpy as np
import time
import imageio.v2 as imageio

import tensorflow as tf
from main import Game
import pygame
import sys
import numpy as np
import time
import imageio.v2 as imageio

import pandas as pd
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
loaded_model = tf.keras.models.load_model('best_model_gen_1.keras')

# Crear una instancia del juego
pygame.init()
pygame.font.init()
game = Game(800, 600)

def preprocess_game_state(game_state, expected_count=5):
    """Preprocesa el estado del juego para que sea compatible con el modelo."""
    player_pos = np.array([game_state['player_position']])  # Posición del jugador
    alien_distances = game_state['alien_x_distances'] + [0] * (expected_count - len(game_state['alien_x_distances']))
    alien_distances = np.array(alien_distances)
    enemy_laser_distances = game_state['enemy_laser_x_distances'] + [0] * (expected_count - len(game_state['enemy_laser_x_distances']))
    enemy_laser_distances = np.array(enemy_laser_distances)
    enemy_direction = np.array([game_state['enemy_direction']])  # Dirección de los enemigos

    input_data = np.concatenate([
        player_pos,
        alien_distances,
        enemy_laser_distances,
        enemy_direction
    ])
    
    return input_data

def play_game(model, game, output_video_path="gameplay_video.mp4"):
    game.reset()
    clock = pygame.time.Clock()
    done = False
    epsilon = 0.01
    total_score = 0

    start_time = time.time()
    time_limit = 180

    # Configuración para la grabación de video
    video_writer = imageio.get_writer(output_video_path, fps=30)

    ALIENLASER = pygame.USEREVENT + 1
    pygame.time.set_timer(ALIENLASER, 800)

    while not done:
        elapsed_time = int(time.time() - start_time)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == ALIENLASER:
                game.alien_shoot()

        game_state = game.get_game_state()
        input_data = preprocess_game_state(game_state)
        input_data = np.nan_to_num(input_data, nan=0, posinf=1e6, neginf=-1e6)
        input_data = np.expand_dims(input_data, axis=0)

        action_probs = model.predict(input_data, verbose=0)[0]

        if np.random.rand() < epsilon:
            action = np.random.choice(len(action_probs))
        else:
            action = np.argmax(action_probs)

        _, reward, done = game.step(action, start_time)
        total_score += reward

        game.screen.fill((30, 30, 30))
        game.run()
        pygame.display.flip()
        clock.tick(60)

        frame = pygame.surfarray.array3d(game.screen)
        frame = np.rot90(frame, 2)
        video_writer.append_data(frame)

    video_writer.close()
    return game.score

# Lista para almacenar los puntajes de cada juego
scores = []

# Ejecutar el juego 30 veces y almacenar los puntajes
for i in range(1):
    video_path = f"gameplay_video{i-1}.mp4"
    score = play_game(loaded_model, game, output_video_path=video_path)
    scores.append(score)
    print(f"Juego {i+1} terminado. Puntaje: {score}")

# Mostrar la lista de puntajes
print("Lista de puntajes:", scores)

