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
    def __init__(self, id):
        obj = Horizons(id=id, location="@sun", epochs=Time("2018-01-01").jd, id_type='id').vectors()
        position = np.multiply([np.double(obj[xi]) for xi in ['x', 'y', 'z']], 149597870700) # gets the x,y,z position of earth and converts to si
        velocity = np.multiply([np.double(obj[vi]) for vi in ['vx', 'vy', 'vz']], (149597870700/(24*3600)))
        self.id = id
        self.position = np.array(position,dtype=np.float)
        self.velocity = np.array(velocity,dtype=np.float)
        self.xpoints = []
        self.ypoints = []
        self.name = np.str(obj['targetname'])
        self.time = 0
        self.orbital_radius = 0


class GenerateRandomBody:
    """
    Generates random solar system
    """
    def __init__(self,id):
        min_distance = 2e9
        max_distance = 1e12
        min_velocity = 0
        max_velocity = 5e4
        self.position = np.array([random.choice([-1,1])*random.randint(min_distance,max_distance),random.choice([-1,1])*random.randint(min_distance,max_distance),0], dtype=np.float)
        self.velocity = np.array([random.choice([-1,1])*random.randint(min_velocity,max_velocity),random.choice([-1,1])*random.randint(min_velocity,max_velocity),0], dtype=np.float)
        self.xpoints = []
        self.ypoints = []
        self.id=id
        self.time = 0
        self.orbital_radius=0