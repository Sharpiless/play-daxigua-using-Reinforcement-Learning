import pygame as pg
from random import randrange
from Fruit import create_fruit
from Game import GameBoard


class Board(GameBoard):
    def __init__(self):
        self.create_time = 2
        self.gravity = (0, 800)
        GameBoard.__init__(self, self.create_time, self.gravity)
        self.init_segment()
        self.setup_collision_handler()
        
    def next_frame(self):
        try:
            if not self.waiting:
                self.count += 1
            self.surface.fill(pg.Color('black'))

            self.space.step(1 / self.FPS)
            self.space.debug_draw(self.draw_options)
            if self.count % (self.FPS * self.create_time) == 0:
                self.i = randrange(1, 5)
                self.current_fruit = create_fruit(
                    self.i, int(self.WIDTH/2), self.init_y - 10)
                self.count = 1
                self.waiting = True

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
                elif event.type == pg.MOUSEBUTTONUP and self.i and self.waiting:
                    x, _ = pg.mouse.get_pos()
                    fruit = create_fruit(self.i, x, self.init_y)
                    self.fruits.append(fruit)
                    ball = self.create_ball(
                        self.space, x, self.init_y, m=fruit.r//10, r=fruit.r-fruit.r % 5, i=self.i)
                    self.balls.append(ball)
                    self.current_fruit = None
                    self.i = None
                    self.waiting = False

            if not self.lock:
                for i, ball in enumerate(self.balls):
                    if ball:
                        angle = ball.body.angle
                        x, y = (int(ball.body.position[0]), int(
                            ball.body.position[1]))
                        self.fruits[i].update_position(x, y, angle)
                        self.fruits[i].draw(self.surface)

            if self.current_fruit:
                self.current_fruit.draw(self.surface)
            pg.draw.aaline(self.surface, (0, 200, 0),
                           (0, self.init_y), (self.WIDTH, self.init_y), 5)
            self.show_score()
            if self.check_fail():
                self.score = 0
                self.last_score = 0
                self.reset()
            pg.display.flip()
            self.clock.tick(self.FPS)

        except Exception as e:
            print(e)
            if len(self.fruits) > len(self.balls):
                seg = len(self.fruits) - len(self.balls)
                self.fruits = self.fruits[:-seg]
            elif len(self.balls) > len(self.fruits):
                seg = len(self.balls) - len(self.fruits)
                self.balls = self.balls[:-seg]

    def run(self):

        while True:
            self.next_frame()


if __name__ == '__main__':

    game = Board()
    game.run()
