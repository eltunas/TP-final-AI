import pygame 

class Laser(pygame.sprite.Sprite):
    def __init__(self, position, speed, screen_height):
        super().__init__()
        self.image = pygame.Surface((4, 20))
        self.is_closest = False  # Flag para indicar si es uno de los más cercanos
        self.update_color()  # Inicializa el color
        self.rect = self.image.get_rect(center=position)
        self.speed = speed
        self.screen_height = screen_height
        self.velocity = [0, -self.speed]
        self.previous_position = self.rect.x, self.rect.y

    def update(self):
        # Actualizar posición
        new_y = self.rect.y + self.speed
        self.velocity[1] = new_y - self.previous_position[1]
        self.rect.y = new_y
        self.previous_position = (self.rect.x, self.rect.y)
        
        # Cambiar el color según el estado de cercanía
        self.update_color()
        
        # Eliminar el láser si sale de la pantalla
        if self.rect.bottom < 0 or self.rect.top > self.screen_height:
            self.kill()

    def update_color(self):
        if self.is_closest:
            self.image.fill((255, 0, 0))  # Rojo para los más cercanos
        else:
            self.image.fill((255, 255, 255))  # Blanco para los demás

    def set_closest_status(self, status):
        self.is_closest = status
