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
from flock_array import FlockArray
import sys

def main():
    pygame.init()
    display = pygame.display.set_mode((1200, 600))
    random.seed(time.time())

    flock = Flock(350)
    arrayFlock = None          # vectorized backend, created lazily on first toggle
    useArray = False           # False -> object/quad-tree backend; True -> array backend
    clock = pygame.time.Clock()

    useTree = True
    showTree = False
    useCohesion = True
    useSeparation = True
    useAlignment = True

    instructions = """
    Click Mouse to add Boid.
    Press 1 to toggle use Quad-tree.
    Press 2 to toggle show Quad-tree.
    Press 3 to toggle Cohesion between Boids.
    Press 4 to toggle Separation between Boids.
    Press 5 to toggle Alignment between Boids.
    Press 6 to toggle the vectorized (NumPy) backend.
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
                mx, my = pygame.mouse.get_pos()
                if useArray:
                    arrayFlock.add_boid(mx, my)
                else:
                    flock.insertBoid((mx, my))
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
                    #toggle Separation
                    useSeparation = not useSeparation
                if event.key == pygame.K_5:
                    #toggle Alignment
                    useAlignment = not useAlignment
                if event.key == pygame.K_6:
                    #toggle the vectorized (NumPy) backend
                    useArray = not useArray
                    if useArray and arrayFlock is None:
                        w, h = pygame.display.get_surface().get_size()
                        arrayFlock = FlockArray(len(flock.flock), w, h)
                if event.key == pygame.K_BACKSPACE:
                    if useArray:
                        arrayFlock.remove_boid()
                    else:
                        flock.removeBoid()


        flockLen = arrayFlock.n if useArray else len(flock.flock)
        backend = "Array(NumPy)" if useArray else "Object/Qtree"
        pygame.display.set_caption(
            "Boids Simulation - Boids: {0} - FPS: {1} - Backend: {2}".format(
                flockLen, int(fps), backend))

        #very basic menu, will implement buttons later.
        states = '\r    Backend: {0},' \
                 ' # of Boids: {1},' \
                 ' Use Qtree: {2},' \
                 ' Show Qtree: {3},' \
                 ' Cohesion: {4},' \
                 ' Separation: {5},' \
                 ' Alignment: {6}     '.\
            format(backend, flockLen, useTree, showTree, useCohesion, useSeparation, useAlignment)

        print(states, end="")

        display.fill((10, 10, 60))
        if useArray:
            mouse = pygame.mouse.get_pos()
            arrayFlock.step(mouse_pos=mouse, useCohesion=useCohesion,
                            useSeparation=useSeparation, useAlignment=useAlignment)
            arrayFlock.draw()
        else:
            flock.draw(useTree, showTree, useCohesion, useSeparation, useAlignment)
        pygame.display.flip()


if __name__ == "__main__":
    main()
