"""
Author: Saad Bhatti
Desc:
This file manages the flock, and applies the Flocking Algorithm to the flock.
"""

import pygame
import pygame.math as m
import random
from boid import Boid
from quad_tree import *
from profileCode import profile


class Flock:
    def __init__(self, popSize, w=0, h=0):
        self.flock = []
        self.createFlock(popSize)

        w, h = pygame.display.get_surface().get_size()
        self.boundary = Rectangle(0, 0, w + 1, h + 1)
        self.quadtree = Quadtree(self.boundary, 2)


    def createFlock(self, popSize):
        for i in range(popSize):
            self.insertBoid()
        return

    def insertBoid(self, pos = None):

        vel = m.Vector2(random.uniform(-10, 10), random.uniform(-10, 10))
        if pos is not None:
            if type(pos) == tuple:
                boid = Boid(m.Vector2(pos[0], pos[1]), vel)
            elif type(pos) == m.Vector2:
                boid = Boid(pos, vel)
        else:
            x, y = pygame.display.get_surface().get_size()
            x = random.randint(0, x)
            y = random.randint(0, y)
            boid = Boid(m.Vector2(x, y), vel)
        self.flock.append(boid)
        return


    def removeBoid(self, boid = None):
        if len(self.flock) == 0:
            print("No boids to remove")
            return

        if boid is not None:
            self.flock.remove(boid)
        else:
            self.flock.pop(0)





    #=================The Rules for Flocking=======================

    def cohesion(self, boid, neighbors):
        #Rule 1: Boids try to fly towards the center of mass of neighbouring boids.
        if len(neighbors) == 0:
            return m.Vector2(0, 0)

        centerOfPos = m.Vector2(0, 0)

        for neighbor in neighbors:
            centerOfPos += neighbor.pos

        centerOfPos /= len(neighbors)
        acc = centerOfPos - boid.pos
        return acc

    def seperation(self, boid, neighbors):
        #Rule 2: Boids try to keep a small distance away from other objects (including other boids).
        if len(neighbors) == 0:
            return m.Vector2(0, 0)

        acc = m.Vector2(0, 0)
        for neighbor in neighbors:
            if boid.pos.distance_to(neighbor.pos) < boid.neighborRadius:
                v = (boid.pos - neighbor.pos)
                r, phi = v.as_polar()
                if r > 0:
                    r = r**-1
                v.from_polar((r, phi))
                acc += v
        return acc

    def alignment(self, boid, neighbors):
        #Rule 3: Boids try to match velocity with nearby boids.
        if len(neighbors) == 0:
            return m.Vector2(0, 0)

        v = m.Vector2(0, 0)
        for neighbor in neighbors:
            v += neighbor.velocity
        v /= len(neighbors)
        acc = v - boid.velocity
        return acc

    def avoidMouse(self, boid):
        #Boids try to avoid/flee mouse position
        mousePos  =  m.Vector2(pygame.mouse.get_pos())
        if not boid.circle.containsPt(Point(mousePos.x, mousePos.y)):
            return m.Vector2(0, 0)

        atVel = boid.pos + boid.velocity
        acc = atVel - mousePos
        if acc.dot(boid.velocity) < 0:
            bx = boid.velocity.x
            ax = acc.x
            sign = lambda a : math.copysign(1, a)

            if not sign(bx) == sign(ax):
                acc.x = 0
            else:
                acc.y = 0

        distTo = boid.pos.distance_to(mousePos)
        if distTo > 0:
            acc *= distTo**-1
        else:
            acc *= 100

        return  acc



    #@profile
    def inNeighboorhood(self, myBoid, useQtree):
        neighbors = []

        if useQtree:
            circleRange = Circle(myBoid.x, myBoid.y, myBoid.neighborRadius)
            neighbors.extend(b for b in self.quadtree.query(circleRange) if b is not myBoid)
        else:
            #O(n^2) algorithm for finding boids in radius
            for boid in self.flock:
                if (myBoid.pos.distance_to(boid.pos) <= myBoid.neighborRadius) and (boid is not myBoid):
                    neighbors.append(boid)
        return neighbors



    def draw(self, useQtree = True, showQtree = True, useCohesion = True, useSeperation = True, useAlignment = True):

        #remake Quad-tree each frame because boids have moved since last frame.
        #remaking Quad-tree doesn't take that much time, so won't affect fps all that much
        self.quadtree.reset()
        self.quadtree.insertPts(self.flock)


        for boid in self.flock:
            neighbors = self.inNeighboorhood(boid, useQtree)

            acc = m.Vector2(0, 0)
            if useCohesion:
                acc += (.004 * self.cohesion(boid, neighbors))
            if useSeperation:
                acc += (1.1 * self.seperation(boid, neighbors))
            if useAlignment:
                acc += (.5 * self.alignment(boid, neighbors))

            acc += (1 * self.avoidMouse(boid))

            #if Boid has no neighbors, it will just wander randomly
            if acc == m.Vector2(0, 0):
                acc += (.3 * m.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)))


            boid.update(acc)
            boid.draw()

        if showQtree:
            self.quadtree.drawBoundaries()


