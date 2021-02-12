import pygame as pg
import pymunk.pygame_util
from Fruit import create_fruit


class GameBoard(object):
    def __init__(self, create_time, gravity):
        self.RES = self.WIDTH, self.HEIGHT = 400, 800
        self.FPS = 50
        self.balls = []
        self.fruits = []

        self.reset()

        self.init_y = int(0.15 * self.HEIGHT)
        self.init_x = int(self.WIDTH / 2)

        pg.init()
        self.surface = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.surface)

        self.space = pymunk.Space()
        self.space.gravity = gravity
        self.create_time = create_time


    def reset(self):
        for ball in self.balls:
            self.space.remove(ball, ball.body)
        del self.fruits
        del self.balls
        self.fruits = []
        self.balls = []
        self.score = 0
        self.last_score = 0
        self.count = 1
        self.lock = False
        self.waiting = False
        self.current_fruit = None
        self.i = None
        self.fail_count = 0
        self.alive = True

    def init_segment(self):
        B1, B2, B3, B4 = (0, 0), (0, self.HEIGHT), (self.WIDTH,
                                                    self.HEIGHT), (self.WIDTH, 0)
        borders = (B1, B2), (B2, B3), (B3, B4)
        for border in borders:
            self.create_segment(*border, 20, self.space, 'darkslategray')

    def setup_collision_handler(self):
        def post_solve_bird_line(arbiter, space, data):
            if not self.lock:
                self.lock = True
                b1, b2 = None, None
                i = arbiter.shapes[0].collision_type + 1
                x1, y1 = arbiter.shapes[0].body.position
                x2, y2 = arbiter.shapes[1].body.position
                if y1 > y2:
                    x, y = x1, y1
                else:
                    x, y = x2, y2
                if arbiter.shapes[0] in self.balls:
                    b1 = self.balls.index(arbiter.shapes[0])
                    space.remove(arbiter.shapes[0], arbiter.shapes[0].body)
                    self.balls.remove(arbiter.shapes[0])
                    fruit1 = self.fruits[b1]
                    self.fruits.remove(fruit1)
                if arbiter.shapes[1] in self.balls:
                    b2 = self.balls.index(arbiter.shapes[1])
                    space.remove(arbiter.shapes[1], arbiter.shapes[1].body)
                    self.balls.remove(arbiter.shapes[1])
                    fruit2 = self.fruits[b2]
                    self.fruits.remove(fruit2)

                fruit = create_fruit(i, x, self.init_y)
                self.fruits.append(fruit)
                ball = self.create_ball(
                    self.space, x, y, m=fruit.r//10, r=fruit.r-1, i=i)
                self.balls.append(ball)
                if i < 11:
                    self.last_score = self.score
                    self.score += i
                elif i == 11:
                    self.last_score = self.score
                    self.score += 100
                self.lock = False

        for i in range(1, 11):
            self.space.add_collision_handler(
                i, i).post_solve = post_solve_bird_line

    def create_ball(self, space, x, y, m=1, r=7, i=1):
        ball_moment = pymunk.moment_for_circle(m, 0, r)
        ball_body = pymunk.Body(m, ball_moment)
        ball_body.position = x, y
        ball_shape = pymunk.Circle(ball_body, r)
        ball_shape.elasticity = 0.3
        ball_shape.friction = 0.6
        ball_shape.collision_type = i
        space.add(ball_body, ball_shape)
        return ball_shape

    def create_segment(self, from_, to_, thickness, space, color):
        segment_shape = pymunk.Segment(
            space.static_body, from_, to_, thickness)
        segment_shape.color = pg.color.THECOLORS[color]
        segment_shape.friction = 0.6
        space.add(segment_shape)

    def show_score(self):
        score_font = pg.font.Font(None, 36)
        score_text = score_font.render(
            'score: {}'.format(str(self.score)), True, (255, 165, 0))
        text_rect = score_text.get_rect()
        text_rect.topleft = [10, 10]
        self.surface.blit(score_text, text_rect)

    def check_fail(self):
        exist = False
        if len(self.balls):
            for i, ball in enumerate(self.balls[:-1]):
                if ball:
                    if int(ball.body.position[1]) < self.init_y:
                        self.fail_count += 1
                        exist = True
                        break
        if exist:
            if self.fail_count > self.FPS*self.create_time:
                self.alive = False
                return True
            return False
        else:
            self.fail_count = 0
            return False

    def run(self):
        pass
