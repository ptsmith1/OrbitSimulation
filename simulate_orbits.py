import scipy.constants as const
import numpy as np


def evolve(body_list, time, dt):
    plots = []
    lines = []
    for b in body_list:
        b.position += b.velocity * dt
        orbital_radius = np.linalg.norm(b.position)
        acc = -((const.G * b.mass * b.position)/(orbital_radius ** 3))
        b.velocity += acc * dt
        b.xpoints.append(b.position[0])
        b.ypoints.append(b.position[1])
        print(b.position, b.velocity, acc)