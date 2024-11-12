import pygame 
from laser import Laser
import time


class Player(pygame.sprite.Sprite):
    def __init__(self, pos, constraint, speed):
        super().__init__()
        self.screen = pygame.display.set_mode((800, 600))
        self.image = pygame.image.load('../graphics/player.png').convert_alpha()
        self.rect = self.image.get_rect(midbottom=pos)
        self.speed = speed
        self.max_x_constraint = constraint
        self.ready = True
        self.laser_time = 0
        self.laser_cooldown = 1000
        self.lasers = pygame.sprite.Group()
        self.laser_sound = pygame.mixer.Sound('../audio/laser.wav')
        self.laser_sound.set_volume(0.5)
        self.last_shot_time = 0
        self.shot_cooldown = 1

    def move(self, direction):
        # Mover al jugador en la dirección especificada
        self.rect.x += direction * self.speed
        # Aplicar las restricciones de la pantalla
        self.constraint()

    def get_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.move(1)
        elif keys[pygame.K_LEFT]:
            self.move(-1)
        if keys[pygame.K_SPACE] and self.ready:
            self.shoot_laser()
            self.ready = False
            self.laser_time = pygame.time.get_ticks()
            self.laser_sound.play()

    def recharge(self):
        if not self.ready:
            current_time = pygame.time.get_ticks()
            if current_time - self.laser_time >= self.laser_cooldown:
                self.ready = True

    def constraint(self):
        if self.rect.left <= 0:
            self.rect.left = 0
        if self.rect.right >= self.max_x_constraint:
            self.rect.right = self.max_x_constraint

    def can_shoot(self):
        # Verificar si el tiempo actual es suficiente para disparar de nuevo
        return time.time() - self.last_shot_time >= self.shot_cooldown

    def shoot_laser(self):
        if self.can_shoot():
            self.last_shot_time = time.time()
            self.lasers.add(Laser(self.rect.center, -8, self.rect.bottom))

    def update(self):
        self.get_input()
        self.constraint()
        self.recharge()
        self.lasers.update()