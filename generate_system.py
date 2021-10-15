import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import scipy.constants as const


class GenerateStockBody:
    """
    Generates our solar system
    """
    def __init__(self, ID):
        obj = Horizons(id=ID, location="@sun", epochs=Time("2018-01-01").jd, id_type='id').vectors()
        position = np.multiply([np.double(obj[xi]) for xi in ['x', 'y', 'z']], 149597870700) # gets the x,y,z position of earth and converts to si
        velocity = np.multiply([np.double(obj[vi]) for vi in ['vx', 'vy', 'vz']], (149597870700/(24*3600)))
        mass_array = [3.3e23,4.87e24,5.97e24,6.43e23,1.898e27,5.68e26,8.68e25,1.02e26,1.46e22]
        self.ID = ID
        self.position = np.array(position,dtype=np.float)
        self.velocity = np.array(velocity,dtype=np.float)
        self.xpoints = []
        self.ypoints = []
        self.name = np.str(obj['targetname'])
        self.time = 0
        self.mass = mass_array[self.ID-1]
        self.orbital_radius = 0


class GenerateRandomBody:
    """
    Generates random solar system
    """
    def __init__(self, ID):
        min_distance = 2e9
        max_distance = 1e12
        min_velocity = 0
        max_velocity = 5e4
        self.position = np.array([random.choice([-1,1])*random.randint(min_distance,max_distance),random.choice([-1,1])*random.randint(min_distance,max_distance),0], dtype=np.float)
        self.velocity = np.array([random.choice([-1,1])*random.randint(min_velocity,max_velocity),random.choice([-1,1])*random.randint(min_velocity,max_velocity),0], dtype=np.float)
        self.acc = np.array([0, 0, 0])
        self.xpoints = []
        self.ypoints = []
        self.ID = ID
        self.time = 0
        self.mass = random.randint(2e24,2e29)
        self.orbital_radius = 0
