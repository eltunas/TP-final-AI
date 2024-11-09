<<<<<<< Updated upstream
import tensorflow as tf
=======
import keras
>>>>>>> Stashed changes
from main import Game  # Asegúrate de importar la clase Game correctamente
import pygame

# Cargar el modelo entrenado
<<<<<<< Updated upstream
loaded_model = tf.keras.models.load_model('best_model.keras')

=======
loaded_model = keras.saving.load_model('best_model.h5')

loaded_model.compile(optimizer=Adam(), 
                     loss=SparseCategoricalCrossentropy(from_logits=True), 
                     metrics=['accuracy'])
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    total_score=0
=======
>>>>>>> Stashed changes

    start_time = time.time()  # Para limitar el tiempo de juego si es necesario
    time_limit = 60  # Puedes ajustar este valor según el tiempo que desees que dure la simulación

    while not done:
        # Calcular el tiempo transcurrido
        elapsed_time = int(time.time() - start_time)

        # Obtener el estado del juego
        game_state = game.get_game_state()  # Asegúrate de que este método existe y devuelve el estado correcto
        input_data = preprocess_game_state(game_state).reshape(1, -1)  # Preprocesar el estado

        # Obtener las probabilidades de las acciones
        action_probs = model.predict(input_data, verbose=0)[0]

        # Decidir si explorar o explotar
        if np.random.rand() < epsilon:
            action = np.random.choice(len(action_probs))  # Exploración (acción aleatoria)
        else:
            action = np.argmax(action_probs)  # Explotación (acción con mayor probabilidad)

<<<<<<< Updated upstream
        print(action)

        # Ejecutar la acción en el juego
        _, reward, done = game.step(action, start_time)
        total_score += reward
=======
        # Ejecutar la acción en el juego
        _, reward, done = game.step(action, start_time)
>>>>>>> Stashed changes

        # Actualizar la pantalla y dibujar el juego
        game.screen.fill((30, 30, 30))  # Fondo oscuro para la pantalla
        game.run()  # Asegúrate de que este método actualiza el estado visual del juego

        # Mostrar el tiempo transcurrido en la parte superior
        timer_text = pygame.font.Font(None, 36).render(f"Tiempo: {elapsed_time}s", True, (255, 255, 255))
        text_rect = timer_text.get_rect(center=(game.screen.get_width() // 2, 20))
        game.screen.blit(timer_text, text_rect)

        pygame.display.flip()  # Actualiza la pantalla
        clock.tick(60)  # Controlar la velocidad del juego (FPS)

        if elapsed_time > time_limit:  # Finaliza después de cierto tiempo
            done = True

<<<<<<< Updated upstream
    print(f"Juego terminado. Su puntaje fue: {total_score}")
=======
    print("Juego terminado")
>>>>>>> Stashed changes


def preprocess_game_state(game_state):
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

play_game(loaded_model, game)