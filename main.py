"""
Author: Saad Bhatti
Desc:
This is an implementation of Craig Reynolds's Flocking Algorithm, which he developed in 1986.
The Bird-oids, Boids, in the flock influence and are influenced by other Boids in its neighborhood.
Based on their neighbors attributes, the Boids adjust their velocity according to 3 core rules.
The first rule is Cohesion. In Cohesion, a Boid will find the center of its neighbors, and will adjust its velocity towards it.
The second rule is Seperation. In Seperation, a Boid will try to maintain a minimum distance from its neighboring Boids.
The third rule is Alignment. In Alignment, a Boid will try to adjust its velocity to match the direction of its neighboring Boids.
The Boids' total acceleration will be a combination of these rules.
The resulting movements and behaviour is very interesting and Boids seem to appear intelligent, that's what makes this algorihtm
one of my favorite algorithms.
"""

import pygame
import random
import time
from flock import Flock
import sys

def main():
    pygame.init()
    display = pygame.display.set_mode((1200, 600))
    random.seed(time.time())

    flock = Flock(250)
    clock = pygame.time.Clock()

    useTree = True
    showTree = False
    useCohesion = True
    useSeperation = True
    useAlignment = True

    instructions = """
    Click Mouse to add Boid.
    Press 1 to toggle use Quad-tree.
    Press 2 to toggle show Quad-tree.
    Press 3 to toggle Cohesion between Boids.
    Press 4 to toggle Seperation between Boids.
    Press 5 to toggle Alignment between Boids.
    """
    print(instructions)


    while True:

        #limit to 30 fps
        clock.tick(30)
        fps = clock.get_fps()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                flock.insertBoid((pygame.mouse.get_pos()))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    #toggle using Quadtree
                    useTree = not useTree
                if event.key == pygame.K_2:
                    #toggle showing tree
                    showTree = not showTree
                if event.key == pygame.K_3:
                    #toggle Cohesion
                    useCohesion = not useCohesion
                if event.key == pygame.K_4:
                    #toggle Seperation
                    useSeperation = not useSeperation
                if event.key == pygame.K_5:
                    #toggle Alignment
                    useAlignment = not useAlignment
                if event.key == pygame.K_BACKSPACE:
                    flock.removeBoid()


        flockLen = len(flock.flock)
        pygame.display.set_caption("Boids Simulation - Boids: {0} - FPS: {1}".format(flockLen, int(fps)))

        #very basic menu, will implement buttons later.
        states = '\r    # of Boids: {0},' \
                 ' Use Qtree: {1},' \
                 ' Show Qtree: {2},' \
                 ' Cohesion: {3},' \
                 ' Seperation: {4},' \
                 ' Alignment: {5}     '.\
            format(flockLen, useTree, showTree, useCohesion, useSeperation, useAlignment)

        print(states, end="")

        display.fill((10, 10, 60))
        flock.draw(useTree, showTree, useCohesion, useSeperation, useAlignment)
        pygame.display.flip()
main()
