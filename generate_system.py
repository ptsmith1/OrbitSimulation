import random
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons


class GenerateBody:
        """
        Generates information about the solar system to be simulated
        """
        def __init__(self,):
            obj = Horizons(id=3, location="@sun", epochs=Time("2018-01-01").jd, id_type='id').vectors()
            position = np.multiply([np.double(obj[xi]) for xi in ['x', 'y', 'z']], 149597870700) # gets the x,y,z position of earth and converts to si
            velocity = np.multiply([np.double(obj[vi]) for vi in ['vx', 'vy', 'vz']], (149597870700/(24*3600)))
            self.radius = 6.371e+6
            self.mass = 1.989e+30
            self.position = np.array(position,dtype=np.float)
            self.velocity = np.array(velocity,dtype=np.float)
            self.xpoints = []
            self.ypoints = []
