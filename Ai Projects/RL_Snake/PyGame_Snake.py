#%% Imports
import pygame as pg
import sys
from pygame.math import Vector2
import random

#%% Init
pg.init()
GREEN = (173, 204, 96)
DARK_GREEN = (43, 51, 24)

class Food:
    def __init__(self, snake_body):
        # self.position = Vector2(5,6) # Vector2 class of Pygame is usefull in this case. 6th column, 7th row.
        self.position = self.generate_random_pos(snake_body)

    def draw(self):
        #? Displace Surface: all the game objects are here. One per game
        #? Regular Surface: here we can draw here. as many Reg. surface as needed.
        #? Rect: used for positioning and collision detection, and for drawing. This is contain the food.
        food_rect = pg.Rect(self.position.x*cell_size, self.position.y*cell_size , cell_size, cell_size) # pg.Rect(x, y , w, h). x-> turn into pix, y-> turn into pix.
        # pg.draw.rect(screen, DARK_GREEN, food_rect ) # (surface, color, rect object)
        screen.blit(food_texture, food_rect.topleft) # (surface, rect object)
    
    def generate_random_cell(self):
        x = random.randint(0, number_of_cells-1)
        y = random.randint(0, number_of_cells-1)
        position = Vector2(x,y)
        return position

    def generate_random_pos(self, snake_body):
        position = self.generate_random_cell()
        #? make sure the generated food location is not coinciding with snake_body location
        while position in snake_body:
            position = self.generate_random_cell()
        return position

class Snake:
    def __init__(self):
        #? snake body = [ head, [tail] ]
        self.body = [Vector2(6,9), Vector2(5,9), Vector2(4,9)]
        self.direction = Vector2(1,0) # to move to the right
        self.add_segment = False

    def draw(self):
        for segment in self.body:
            segment_rect = (segment.x*cell_size, segment.y*cell_size, cell_size, cell_size)
            pg.draw.rect(screen, DARK_GREEN, segment_rect, 0, 7) # (surface, color, rect object, <filled with color>, <border>)
    
    def update(self):
        self.body.insert(0, self.body[0]+self.direction) # increase the snake body in it's head direction. at 0th index add direction to the head. 
        if self.add_segment == True:
            self.add_segment = False
        else:
            self.body = self.body[:-1] # removed one from the tail. 
    
    def reset(self):
        self.body = [Vector2(6,9), Vector2(5,9), Vector2(4,9)]

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food  = Food(self.snake.body)
        self.state = "RUNNING"

    def draw(self):
        self.food.draw()
        self.snake.draw()

    def update(self):
        if self.state == "RUNNING":
            self.snake.update()
            self.check_collision_with_food()
            self.check_collision_with_edges()
            self.check_collision_with_tail()
    
    def check_collision_with_food(self):
        if self.snake.body[0] == self.food.position:
            self.food.position = self.food.generate_random_pos(self.snake.body)
            self.snake.add_segment = True
    
    def check_collision_with_edges(self):
        if self.snake.body[0].x == number_of_cells or self.snake.body[0].x == -1: # snake passed the right edge or left edge
            self.game_over()
        if self.snake.body[0].y == number_of_cells or self.snake.body[0].y == -1: # snake passed the bottom or top
            self.game_over()
    
    def check_collision_with_tail(self):
        headless_body = self.snake.body[1:]
        if self.snake.body[0] in headless_body:
            self.game_over()

    def game_over(self):
        print("Game Over")
        self.snake.reset()
        self.food.generate_random_pos(self.snake.body)
        self.state = "STOPPED"

#* Canvas
cell_size = 30
number_of_cells = 25
OFFSET = 75
OFFSET_BORDER = 5

screen = pg.display.set_mode((2*OFFSET + cell_size*number_of_cells, 2*OFFSET + cell_size*number_of_cells)) # origin-> top-left.
pg.display.set_caption("Retro Snake")

clock = pg.time.Clock()


game = Game()
food_texture = pg.image.load("Ai Projects/RL_Snake/Assets/food_star.png").convert_alpha() # keeps transparency
food_texture = pg.transform.smoothscale(food_texture, (cell_size, cell_size))

#%% Custom Event
#? lets trigger a custom event at slower pace than game loop (60Hz).
SNAKE_UPDATE = pg.USEREVENT # create special PyGame user event
pg.time.set_timer(SNAKE_UPDATE, 200) # create timer that will trigger SNAKE_UPDATE event every 200 ms.



#%% Game Loop
#? Event Handling, Updating positions, Drawing Objects.
while True:
    for event in pg.event.get():
        #? Check for any event happend since the last time this while loop is executed
        if event.type == SNAKE_UPDATE:
            game.update()
        
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        
        #? Make sure snake doesn't turn 180 degree !
        if event.type == pg.KEYDOWN:
            if game.state == "STOPPED":
                game.state = "RUNNING"

            if event.key == pg.K_UP and game.snake.direction != Vector2(0,1): #? if key=UP and snake is not moving downwards.
                game.snake.direction = Vector2(0, -1) # move up
            if event.key == pg.K_DOWN and game.snake.direction != Vector2(0,-1): #? if key=DOWN and snake is not moving downwards.
                game.snake.direction = Vector2(0, 1)  # move down
            if event.key == pg.K_RIGHT and game.snake.direction != Vector2(-1, 0): #? if key=RIGHT and snake is not moving left.
                game.snake.direction = Vector2(1, 0)  # move right
            if event.key == pg.K_LEFT and game.snake.direction != Vector2(1, 0): #? if key=LEFT and snake is not moving right.
                game.snake.direction = Vector2(-1, 0) # move left

    # Drawing
    screen.fill(GREEN)  # Fill the display surface.
    pg.draw.rect(screen, DARK_GREEN, (OFFSET-5, OFFSET-5, cell_size*number_of_cells+10, cell_size*number_of_cells+10), 5) # (surface, color, rect, border_size)
    game.draw()

    pg.display.update() # takes all the changes and draws updates
    clock.tick(60) # FPS. everything insider while loop will run 60 times every second.

