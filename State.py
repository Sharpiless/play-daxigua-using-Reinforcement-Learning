import pygame as pg
from random import randrange
from Fruit import create_fruit
from Game import GameBoard


class AI_Board(GameBoard):
    def __init__(self):
        self.create_time = 0.5
        self.gravity = (0, 4000)
        GameBoard.__init__(self, self.create_time, self.gravity)
        self.action_num = 16
        self.init_segment()
        self.setup_collision_handler()

    def decode_action(self, action):

        seg = (self.WIDTH - 40) // self.action_num
        x = action * seg + 20
        print('-[INFO] Drop down at x =', x)

        return x

    def next_frame(self, action=None):
        try:
            reward = 0
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
            if not action is None and self.i and self.waiting:
                x = self.decode_action(action)
                fruit = create_fruit(self.i, x, self.init_y)
                self.fruits.append(fruit)
                ball = self.create_ball(
                    self.space, x, self.init_y, m=fruit.r//10, r=fruit.r-fruit.r % 5, i=self.i)
                self.balls.append(ball)
                self.current_fruit = None
                self.i = None
                self.waiting = False

            reward = self.score - self.last_score
            if reward > 0:
                self.last_score = self.score

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
            image = pg.surfarray.array3d(pg.display.get_surface())

        except Exception as e:
            print(e)
            if len(self.fruits) > len(self.balls):
                seg = len(self.fruits) - len(self.balls)
                self.fruits = self.fruits[:-seg]
            elif len(self.balls) > len(self.fruits):
                seg = len(self.balls) - len(self.fruits)
                self.balls = self.balls[:-seg]

        return image, self.score, reward, self.alive

    def next(self, action=None):
        _, _, reward, _ = self.next_frame(action=action)
        for _ in range(self.FPS * 3):
            _, _, nreward, _ = self.next_frame()
            reward += nreward
        image, _, nreward, _ = self.next_frame()
        reward += nreward
        if reward == 0:
            reward = -self.i
        return image, self.score, reward, self.alive

    def run(self):

        while True:
            action = randrange(0, self.action_num)
            print('action:', action)
            _, score, reward, alive = self.next(action=action)
            print('score:{} reward:{} alive:{}'.format(score, reward, alive))


if __name__ == '__main__':

    game = AI_Board()
    game.run()
