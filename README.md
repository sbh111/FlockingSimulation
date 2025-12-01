# FlockingSimulation
An implementation of Craig Reynolds's Boids flocking algorithm in Python.

To run this project, clone this repo with **git clone url**.  
Then download the required pygame package with **pip install -r requirements.txt"**.  
Then run it with **python main.py**.  

# Improvements
The Quad-Tree data structure is not as effiecent as i'd hoped, it's infact... slower than the naive O(n^2) implementation.
So i was thinking of other ways to speed up implementation. Options include:
 - Spatial Partitioning
 - GPU Paralellization using CuPy


# Demo
![](res/demo.gif)

