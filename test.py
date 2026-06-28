"""
Desc:
just some test code used to test Quad-tree.
"""
import pygame
import random
import time
from quad_tree import *


def main():
    pygame.init()
    display = pygame.display.set_mode((600, 600))
    random.seed(time.time())

    clock = pygame.time.Clock()

    pts = []
    qtree = Quadtree(Rectangle(0, 0, 601, 601), 150)
    for i in range(5000):
        pts.append((random.randint(0, 600), random.randint(0, 600)))


    while True:
        clock.tick()
        fps = clock.get_fps()
        pygame.display.set_caption("Test - FPS: {}".format(int(fps)))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pts.append((pygame.mouse.get_pos()))
            if event.type == pygame.KEYDOWN:
                if pygame.K_BACKSPACE and len(pts) > 0:
                    pts.pop(0)


        pts.append((random.randint(0, 600), random.randint(0, 600)))
        pts.pop(0)

        display.fill((0, 0, 50))
        for pt in pts:
            pygame.draw.circle(display, (255, 0, 0), pt, 4)

        qtree.reset()
        qtree.insertPts(pts)

        mousePos = pygame.mouse.get_pos()
        rect = Rectangle(mousePos[0]-50, mousePos[1]-50, 100, 100)
        circle = Circle(mousePos[0], mousePos[1], 100)

        containedPts = []
        containedPts = qtree.query(circle)
        for pt in containedPts:
            pygame.draw.circle(display, (255, 255, 255), pt, 4)
        containedPts = qtree.query(rect)
        for pt in containedPts:
            pygame.draw.circle(display, (255, 0, 255), pt, 4)

        pygame.draw.rect(display, (0, 255, 0), rect.getRect(), 2)
        pygame.draw.circle(display, (0, 255, 0), mousePos, circle.radius, 2)

        qtree.drawBoundaries()
        pygame.display.flip()


if __name__ == "__main__":
    main()