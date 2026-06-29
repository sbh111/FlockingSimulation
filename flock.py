"""
Author: Saad Bhatti
Desc:
This file manages the flock, and applies the Flocking Algorithm to the flock.
"""

import pygame
import pygame.math as m
import math
import random
from boid import Boid
from quad_tree import *
from profileCode import profile


# Separation weight is inversely proportional to the flock size: the per-frame
# weight is SEPARATION_K / (number of boids). SEPARATION_K is calibrated so the
# weight equals the previous fixed 1.1 at the default 350-boid flock
# (1.1 * 350 = 385), so behaviour is unchanged at startup and only weakens as
# the flock grows (and strengthens as it shrinks).
SEPARATION_K = 385.0


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

    def applyRules(self, boid, neighbors, useCohesion, useSeperation, useAlignment):
        #Computes the combined acceleration from the three core flocking rules in a
        #single pass over the neighbor list, instead of iterating it once per rule.
        #  Rule 1 (Cohesion):   fly towards the center of mass of neighbouring boids.
        #  Rule 2 (Seperation): keep a small distance away from other boids.
        #  Rule 3 (Alignment):  match velocity with nearby boids.
        n = len(neighbors)
        if n == 0:
            return m.Vector2(0, 0)

        sumPos = m.Vector2(0, 0)   #for cohesion
        sumVel = m.Vector2(0, 0)   #for alignment
        sepAcc = m.Vector2(0, 0)   #for seperation

        for neighbor in neighbors:
            if useCohesion:
                sumPos += neighbor.pos
            if useAlignment:
                sumVel += neighbor.velocity
            if useSeperation:
                #every neighbor is already within neighborRadius (that's how it was
                #found), so no distance filter is needed here. Weight by 1/distance.
                v = (boid.pos - neighbor.pos)
                r, phi = v.as_polar()
                if r > 0:
                    r = r**-1
                v.from_polar((r, phi))
                sepAcc += v

        acc = m.Vector2(0, 0)
        if useCohesion:
            acc += .004 * ((sumPos / n) - boid.pos)
        if useSeperation:
            #separation weight scales inversely with the total flock size
            sepFactor = SEPARATION_K / len(self.flock)
            acc += sepFactor * sepAcc
        if useAlignment:
            acc += .5 * ((sumVel / n) - boid.velocity)
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

            acc = self.applyRules(boid, neighbors, useCohesion, useSeperation, useAlignment)
            acc += (1 * self.avoidMouse(boid))

            #if Boid has no neighbors, it will just wander randomly
            if acc == m.Vector2(0, 0):
                acc += (.3 * m.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)))


            boid.update(acc)
            boid.draw()

        if showQtree:
            self.quadtree.drawBoundaries()


