
#This file holds commonly used shapes and classes.
import math
import pygame


class Point:
    def __init__(self, x, y, data = None):
        self.x = x
        self.y = y
        self.data = data

    @staticmethod
    def distToLine(px, py, lx1, ly1, lx2, ly2):
        #calculates euclidian distance.
        A = px - lx1
        B = py - ly1
        C = lx2 -lx1
        D = ly2 - ly1

        dot = A * C + B * D
        lenSq = C**2 + D**2
        param = -1

        if not lenSq == 0:
            param = dot / lenSq

        if param < 0:
            xx = lx1
            yy = ly1
        elif param > 1:
            xx = lx2
            yy = ly2
        else:
            xx = lx1 + param * C
            yy = ly1 + param * D

        dx = px - xx
        dy = py - yy
        return math.sqrt(dx**2 + dy**2)


class Circle:
    def __init__(self, x, y, r):
        self.radius = r
        self.x = x
        self.y = y

    def containsPt(self, point):
        dx = self.x - point.x
        dy = self.y - point.y
        return dx * dx + dy * dy <= self.radius * self.radius

    def containsPts(self, points):
        containsPts = []
        for pt in points:
            if self.containsPt(pt):
                containsPts.append(pt)
        return containsPts







class Rectangle:
    def __init__(self, x, y, w, h):
        #x and y are coords for top left
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def intersects(self, boundary):
        #check to see if rect intersects a boundary.
        #boundary can be a Rect or Circle

        # this rects left, right, top, bottom
        l0 = self.x
        r0 = l0 + self.w
        t0 = self.y
        b0 = t0 + self.h

        if type(boundary) == Rectangle:
            #boundary's left, right, top, bottom
            l1 = boundary.x
            r1 = l1 + boundary.w
            t1 = boundary.y
            b1 = t1 + boundary.h

            return not (r0 < l1 or r1 < l0 or
                        t0 > b1 or t1 > b0
                        )

        if type(boundary) == Circle:
            cx = boundary.x
            cy = boundary.y
            cr = boundary.radius
            # Closest point on the AABB to the circle center
            closest_x = max(l0, min(cx, r0))
            closest_y = max(t0, min(cy, b0))
            dx = cx - closest_x
            dy = cy - closest_y
            return dx * dx + dy * dy <= cr * cr
        raise Exception("boundary should be a Rectangle or Circle type")


    def containsPts(self, points):
        containsPts = []
        for pt in points:
            if self.containsPt(pt.x, pt.y):
                containsPts.append(pt)
        return containsPts

    def containsPt(self, x, y):
        return (
                x >= self.x and
                x < self.x + self.w and
                y >= self.y and
                y < self.y + self.h
        )

    def getRect(self):
        return pygame.Rect(int(self.x), int(self.y), int(self.w), int(self.h))

