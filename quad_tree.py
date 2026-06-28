"""
Author: Saad Bhatti
Desc:
This data structure is called a Quad-tree. A Quad-tree is used store and organize to positional data
in its leaves. When a point is inserted, the Quad-tree will put it into the node which corresponds with the Quadrant/Region
it was in. The nodes/children of the Quad-tree can only hold upto the specified maximum capacity.
When a leaf reaches its max capacity, it will subdivide into 4 more nodes, and reinsert the points into their repectivr quadrants.
The main benifit of a Quad-tree is an O(NLogN) query. If you want to find points in a boundary, the Quad-tree will only
search the regions that intersect with the boundary. In a naive query, you would have to iterate over all the
points in the map, which would be an O(N^2) procedure. With using a Quad-tree query, the computational complexity of a query
goes down from O(N^2) to O(NlogN).
"""

import pygame
from shapes import *


class Node:
    def __init__(self, rectBoundary, points=None):
        self.rectBoundary = rectBoundary
        self.points = points if points is not None else []
        self.children = []
        self.isSubdivided = False



class Quadtree:
    def __init__(self, rectBoundary, capacity = 4, maxheight = 5):
        self.root = Node(rectBoundary)
        self.capacity = capacity
        self.maxheight = maxheight

    def subdivide(self):
        recursiveSubdivide(self.root, self.capacity, self.maxheight)

    def insert(self, point):
        recursiveInsert(self.root, self.capacity, point, self.maxheight)

    def insertPts(self, objects):
        for object in objects:
            if isinstance(object, Point):
                recursiveInsert(self.root, self.capacity, object, self.maxheight)
            elif type(object) == tuple:
                recursiveInsert(self.root, self.capacity, Point(object[0], object[1], object), self.maxheight)
            else:
                recursiveInsert(self.root, self.capacity, Point(object.x, object.y, object), self.maxheight)

    def query(self, range, isIterative = False):
        #Passing in a list so no time/space wasted in copying
        dataList = []
        if isIterative:
            iterativeQuery(self.root, dataList, range)
        else:
            recursiveQuery(self.root, dataList, range)
        return [pt.data for pt in dataList]


    def drawBoundaries(self):
        screen = pygame.display.get_surface()
        recursiveDrawBoundaries(self.root, screen)

    def reset(self):
        self.root.points = []
        self.root.children = []
        self.root.isSubdivided = False



#Quadtree helpers

def recursiveQuery(node, dataList, range):
    if not node.rectBoundary.intersects(range):
        return
    if node.isSubdivided:
        for child in node.children:
            recursiveQuery(child, dataList, range)
        return
    pts = range.containsPts(node.points)
    dataList.extend(pts)


def iterativeQuery(root, dataList, range):
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.rectBoundary.intersects(range):
            continue
        if node.isSubdivided:
            stack.extend(node.children)
        else:
            pts = range.containsPts(node.points)
            dataList.extend(pts)






def recursiveDrawBoundaries(node, screen):
    #Just draw a pygame rect using the node's getRect method
    pygame.draw.rect(screen, (120, 120, 120), node.rectBoundary.getRect(), 1)
    if len(node.children) > 0:
        for child in node.children:
            recursiveDrawBoundaries(child, screen)
    return


def recursiveInsert(node, capacity, point, currHeight):
    if not node.rectBoundary.containsPt(point.x, point.y):
        return False

    if currHeight == 0:
        node.points.append(point)
        return True

    if node.isSubdivided:
        for child in node.children:
            if recursiveInsert(child, capacity, point, currHeight - 1):
                return True
        return False

    node.points.append(point)
    if len(node.points) > capacity:
        recursiveSubdivide(node, capacity, currHeight)
    return True


def recursiveSubdivide(node, capacity, currHeight):
    # Stop if within capacity, or if we've hit the depth limit. The depth
    # limit is what guarantees termination when points are coincident
    # (e.g. boids at the exact same position), since those always fall into
    # the same child quadrant and would otherwise subdivide forever.
    if len(node.points) <= capacity or currHeight == 0:
        return

    newW = node.rectBoundary.w / 2
    newH = node.rectBoundary.h / 2
    x = node.rectBoundary.x
    y = node.rectBoundary.y

    topLeftRect = Rectangle(x, y, newW, newH)
    pts = topLeftRect.containsPts(node.points)
    topLeft = Node(topLeftRect, pts)
    recursiveSubdivide(topLeft, capacity, currHeight - 1)

    topRightRect = Rectangle(x + newW, y, newW, newH)
    pts = topRightRect.containsPts(node.points)
    topRight = Node(topRightRect, pts)
    recursiveSubdivide(topRight, capacity, currHeight - 1)

    botLeftRect = Rectangle(x, y + newH, newW, newH)
    pts = botLeftRect.containsPts(node.points)
    botLeft = Node(botLeftRect, pts)
    recursiveSubdivide(botLeft, capacity, currHeight - 1)

    botRightRect = Rectangle(x + newW, y + newH, newW, newH)
    pts = botRightRect.containsPts(node.points)
    botRight = Node(botRightRect, pts)
    recursiveSubdivide(botRight, capacity, currHeight - 1)

    node.children = [topLeft, topRight, botLeft, botRight]
    node.points = []
    node.isSubdivided = True
    return






