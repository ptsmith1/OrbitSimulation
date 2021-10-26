import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import scipy.constants as const

class GenerateBody:
    """
    Parent class for orbital bodies
    """
    def __init__(self):
        self.xpoints = []
        self.ypoints = []
        self.orbital_radii = []
        self.time = 0
        self.orbital_radius = 0
        self.position = 0
        self.velocity = 0
        self.k = 0
        self.mass = 0

    def calc_energy_conservation(self, sun_mass):
        orbital_distance = np.linalg.norm(self.position)
        velocity_magnitude = np.linalg.norm(self.velocity)
        self.k = -(const.G * self.mass * sun_mass)/orbital_distance + 0.5 * self.mass * velocity_magnitude * velocity_magnitude

class GenerateStockBody(GenerateBody):
    """
    Generates our solar system
    """
    def __init__(self, ID, sun_mass):
        GenerateBody.__init__(self)
        obj = Horizons(id=ID, location="@sun", epochs=Time("2018-01-01").jd, id_type='id').vectors()
        position = np.multiply([np.double(obj[xi]) for xi in ['x', 'y', 'z']], 149597870700) # gets the x,y,z position of earth and converts to si
        velocity = np.multiply([np.double(obj[vi]) for vi in ['vx', 'vy', 'vz']], (149597870700/(24*3600)))
        mass_array = [3.3e23,4.87e24,5.97e24,6.43e23,1.898e27,5.68e26,8.68e25,1.02e26,1.46e22]
        self.ID = ID
        self.position = np.array(position,dtype=np.float)
        self.velocity = np.array(velocity,dtype=np.float)
        self.name = np.str(obj['targetname'])
        self.mass = mass_array[self.ID-1]
        self.calc_energy_conservation(sun_mass)


class GenerateRandomBody(GenerateBody):
    """
    Generates random solar system
    """
    def __init__(self, ID, sun_mass):
        GenerateBody.__init__(self)
        min_distance = 2e9
        max_distance = 2e12
        min_velocity = 5e2
        max_velocity = 5e3
        self.position = np.array([random.choice([-1,1])*random.randint(min_distance,max_distance),random.choice([-1,1])*random.randint(min_distance,max_distance),0], dtype=np.float)
        self.velocity = np.array([random.choice([-1,1])*random.randint(min_velocity,max_velocity),random.choice([-1,1])*random.randint(min_velocity,max_velocity),0], dtype=np.float)
        self.acc = np.array([0, 0, 0])
        self.ID = ID
        self.mass = random.randint(2e23,2e27)
        self.calc_energy_conservation(sun_mass)


class GenerateDebugBody(GenerateBody):
    """
    Generates random solar system
    """
    def __init__(self, ID, sun_mass):
        GenerateBody.__init__(self)
        x_distance = 1e12
        y_distance = 1e12
        x_velocity = 0
        y_velocity = 0
        self.position = np.array([x_distance,y_distance,0], dtype=np.float)
        self.velocity = np.array([x_velocity,y_velocity,0], dtype=np.float)
        self.acc = np.array([0, 0, 0])
        self.ID = ID
        self.mass = 2e23
        self.calc_energy_conservation(sun_mass)