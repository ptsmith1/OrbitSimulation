import random
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import scipy.constants as const

class SystemEvolve:
    """

    """
    def __init__(self):
        self.time = 0
        self.dt = 60 * 60 * 24

    def evolve(self):
        plots = []
        lines = []
        self.position += self.velocity * self.dt
        orbital_radius = np.linalg.norm(self.position)
        acc = -((const.G * self.mass * self.position) / (orbital_radius ** 3))
        self.velocity += acc * self.dt
        self.xpoints.append(self.position[0])
        self.ypoints.append(self.position[1])
        print(self.position, self.velocity, acc)
