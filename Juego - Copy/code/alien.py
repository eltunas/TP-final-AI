import pygame

class Alien(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        super().__init__()
        file_path = '../graphics/' + color + '.png'
        self.image = pygame.image.load(file_path).convert_alpha()
        self.rect = self.image.get_rect(topleft=(x, y))
        
        # Valor de los aliens según color
        if color == 'red': 
            self.value = 100
        elif color == 'green': 
            self.value = 200
        else: 
            self.value = 300
        
        # Inicializar velocidad en x y y
        self.velocity = [0, 0]
        self.previous_position = self.rect.x, self.rect.y

    def update(self, direction):
        # Calcular nueva posición y actualizar
        new_x = self.rect.x + direction
        self.velocity[0] = new_x - self.previous_position[0]  # Cambio en x
        self.velocity[1] = self.rect.y - self.previous_position[1]  # Cambio en y (sin cambios para aliens)
        
        self.rect.x = new_x
        self.previous_position = (self.rect.x, self.rect.y)

class Extra(pygame.sprite.Sprite):
	def __init__(self,side,screen_width):
		super().__init__()
		self.image = pygame.image.load('../graphics/extra.png').convert_alpha()
		
		if side == 'right':
			x = screen_width + 50
			self.speed = - 3
		else:
			x = -50
			self.speed = 3

		self.rect = self.image.get_rect(topleft = (x,80))

	def update(self):
		self.rect.x += self.speed