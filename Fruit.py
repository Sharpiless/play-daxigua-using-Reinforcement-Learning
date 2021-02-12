import pygame as pg


def create_fruit(type, x, y):
    fruit = None
    if type == 1:
        fruit = PT(x, y)
    elif type == 2:
        fruit = YT(x, y)
    elif type == 3:
        fruit = JZ(x, y)
    elif type == 4:
        fruit = NM(x, y)
    elif type == 5:
        fruit = MHT(x, y)
    elif type == 6:
        fruit = XHS(x, y)
    elif type == 7:
        fruit = TZ(x, y)
    elif type == 8:
        fruit = BL(x, y)
    elif type == 9:
        fruit = YZ(x, y)
    elif type == 10:
        fruit = XG(x, y)
    elif type == 11:
        fruit = DXG(x, y)
    return fruit


class Fruit():
    def __init__(self, x, y):

        self.load_images()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.angle_degree = 0

    def load_images(self):
        pass

    def update_position(self, x, y, angle_degree=0):
        self.rect.x = x - self.r
        self.rect.y = y - self.r
        self.angle_degree = angle_degree
        # self.image = pg.transform.rotate(self.image, self.angle_degree)

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class PT(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 10
        self.type = 1
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/01.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class YT(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 15
        self.type = 2
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/02.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class JZ(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 21
        self.type = 3
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/03.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class NM(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 23
        self.type = 4
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/04.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class MHT(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 29
        self.type = 5
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/05.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class XHS(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 35
        self.type = 6
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/06.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class TZ(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 37
        self.type = 7
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/07.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class BL(Fruit):
    def __init__(self, x, y):
        self.r = 2 * 50
        self.type = 8
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/08.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class YZ(Fruit):

    def __init__(self, x, y):
        self.r = 2 * 59
        self.type = 9
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/09.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class XG(Fruit):

    def __init__(self, x, y):
        self.r = 2 * 60
        self.type = 10
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/10.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class DXG(Fruit):

    def __init__(self, x, y):
        self.r = 2 * 78
        self.type = 11
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/11.png')
        self.image = pg.transform.smoothscale(self.image, self.size)
