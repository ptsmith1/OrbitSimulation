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
        self.dt = 60 * 60 * 24
        self.mass = 1.989e30

    def evolve(self, planet):
        planet.position += planet.velocity * self.dt
        planet.orbital_radius = np.linalg.norm(planet.position)
        acc = -((const.G * self.mass * planet.position) / (planet.orbital_radius ** 3))
        planet.velocity += acc * self.dt
        planet.xpoints.append(planet.position[0])
        planet.ypoints.append(planet.position[1])
        planet.time += self.dt
        if len(planet.xpoints) >= 100:
            planet.xpoints.pop(0)
            planet.ypoints.pop(0)
    #    if planet.id == 6: print(planet.position, planet.velocity, acc)
        #if planet.id == 6: print(planet.time/(self.dt*365))
