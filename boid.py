"""
Author: Saad Bhatti
Desc:
This Boid Class holds the methods and attributes of the agent.
Each boid has an x, y and velocity. The velocity is influenced by the acceleration calculated
in the Flock class.
"""
import pygame
import pygame.math as m
import random
from shapes import *

class Boid:
    def __init__(self, pos = m.Vector2(0, 0), vel = m.Vector2(0, 0)):

        self.pos = pos
        self.x = pos.x
        self.y = pos.y
        self.velocity = vel

        self.s = int(random.uniform(1,5))
        r = int(random.gauss(30, 20))
        self.neighborRadius =  r if self.s * 10 < r else self.s * 10
        self.circle = Circle(self.x, self.y, self.neighborRadius)
        self.maxSpeed = abs(random.gauss(20,5))
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    def limitSpeed(self):
        if self.velocity.length() > self.maxSpeed:
            self.velocity.scale_to_length(self.maxSpeed)

    def wrapAround(self):
        x, y = self.pos
        w, h = pygame.display.get_surface().get_size()

        if x < 0:
            x = w
        elif x > w:
            x = 0

        if y < 0:
            y = h
        elif y > h:
            y = 0
        self.pos.update(x, y)

    def draw(self):
        screen = pygame.display.get_surface()
        #pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.s)

        vec = self.pos + 15*self.velocity.normalize()
        end_pos = [vec.x, vec.y]
        start_pos = [int(self.x), int(self.y)]
        pygame.draw.line(screen, self.color, start_pos, end_pos, self.s)
        #pygame.draw.circle(screen, (10, 10, 90), (int(self.x), int(self.y)), self.neighborRadius, 1)



    def update(self, acc):
        self.velocity += acc
        self.limitSpeed()
        self.pos += self.velocity
        self.wrapAround()

        self.x = self.pos.x
        self.y = self.pos.y
        self.circle.x = self.x
        self.circle.y = self.y






