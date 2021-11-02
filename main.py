import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import math
import cProfile, pstats, io
from timeit import default_timer as timer
from numba import vectorize, cuda, jit, njit, guvectorize
from generate_system import *


def main():
    start = time.time_ns()
    sim = Simulation()
    sim.run_sim()
    end = time.time_ns()
    if sim.plot_data:
        plot = Plotting(sim)
        plot.create_plot(sim)
        plot.start_anim(sim)
        # if len(planet.xpoints) >= self.days_to_plot:
        #     planet.xpoints.pop(0)
        #     planet.ypoints.pop(0)
    print("Run time: ", (end - start)/1e9, "seconds")


class Simulation:
    """
    Holds the top level methods and parameters
    """
    def __init__(self):
        """Simulation parameters"""
        self.random_planets = False  # if false use our solar system
        self.stock_planets = False
        self.debugging = True
        self.planet_count = 5  # planets to create, max 9 if using our solar system
        self.dt = int(3600 * 24)  # in seconds
        self.run_time = 30 # in years
        self.sun_mass = 1.989e30
        self.minimum_interaction_distance = 1e8  # to stop planets flying off at high velocities due to a timestep placing a body very close to another (this is a temporary fudge)
        self.parallel_speedup_on = True # choose whether to use parallel speedup
        """Plotting parameters"""
        self.plot_data = True  # choose whether to plot the positions of planets
        self.plot_labels = True
        self.save_anim = False
        self.axes_limit = 3000e9  # size of plot window in meters
        self.plot_interval = 25  # in days
        self.days_to_plot = 100  # the number of days for which the position of each planet is plotted
        self.delete_distance = 30  # the distance from the Sun at which a planet is deleted in AU
        self.colours = ['gray', 'orange', 'blue', 'chocolate','brown','hotpink','black','green','violet']  # line colours
        """--------------------"""
        self.planets = []
        #all calculations are done through these arrays, each planet then collects its data from these arrays for the purpose of data plotting
        self.mass_array = np.empty([self.planet_count,3])
        self.pos_array = np.empty([self.planet_count,3])
        self.vel_array = np.empty([self.planet_count,3])
        self.acc_array = np.empty([self.planet_count,3])
        self.times = []
        if self.stock_planets:
            self.planet_count = 10
        self.create_planets()

    def create_planets(self):
        self.planets.append(GenerateStar(0, self.sun_mass))
        for i in range(self.planet_count -1):
            """
            Generates the planets with random velocity/positions or based off our solar system
            """
            if self.random_planets:
                self.planets.append(GenerateRandomBody(i + 1, self.sun_mass))
            elif self.debugging:
                self.planets.append(GenerateDebugBody(i + 1, self.sun_mass))
            elif self.stock_planets:
                self.planets.append(GenerateStockBody(i + 1))
        self.mass_array = np.asarray([self.planets[i].mass for i in range(self.planet_count)], dtype=np.float64)
        self.pos_array = np.stack([self.planets[i].position for i in range(self.planet_count)], axis=0)
        self.vel_array = np.stack([self.planets[i].velocity for i in range(self.planet_count)], axis=0)
        self.acc_array = np.zeros_like(self.pos_array, dtype=np.float64)
        gpe_array = self.get_gpe()
        for n, planet in enumerate(self.planets):
            planet.k = gpe_array[n] + 0.5 * planet.mass * np.linalg.norm(planet.velocity) * np.linalg.norm(planet.velocity)

    def run_sim(self):
        for i in range(self.run_time * 365 * ((3600 * 24)//self.dt)):
            self.times.append(self.planets[0].time)
            self.evolve(self.planets)
            if i % 25 == 0:
                # prints every 25 timesteps
                print("Year:", self.planets[1].time / (3600 * 24 * 365))
                print("ID:", self.planets[1].ID, "Position:", self.planets[1].position, "Velocity", self.planets[1].velocity, "Acceleration", self.planets[1].acc)
        print("Year:", self.planets[1].time / (3600 * 24 * 365))
        print("ID:", self.planets[1].ID, "Position:", self.planets[1].position, "Velocity", self.planets[1].velocity,"Acceleration", self.planets[1].acc)

    def evolve(self, planets):
        """
        Updates the location of each planet based on its acceleration due to the gravitation field of the sun and other planets
        """
        self.acc_array = np.zeros_like(self.pos_array, dtype=np.float64)
        self.pos_array = np.add(self.pos_array, self.vel_array * self.dt)
        if self.parallel_speedup_on:
            self.calc_acc_parallel(self.pos_array, self.pos_array, self.mass_array, np.int64(self.minimum_interaction_distance), self.acc_array)
        else:
            self.calc_acc()
        self.vel_array = np.add(self.vel_array, self.acc_array * self.dt)
        velocity_corrections = self.check_energy_conservation()
        for i in range(self.planet_count):
            self.vel_array[i] = np.multiply(self.vel_array[i], velocity_corrections[i])

        for n, planet in enumerate(planets):
            # creates data for outputting/visualisation on each planet
            planet.position = self.pos_array[n]
            planet.orbital_radius = np.linalg.norm(planet.position)
            planet.acc = self.acc_array[n]
            planet.velocity = self.vel_array[n]
            planet.xpoints.append(planet.position[0])
            planet.ypoints.append(planet.position[1])
            planet.orbital_radii.append(planet.orbital_radius)
            planet.time += self.dt

    def calc_acc(self):
        """
        Calculates the accelreration between each planet and every other planet, makes sure to not calculate the
        attraction between eg p1 to p2 and between p2 to p1, instead using the first calculated value and changing
        the sign.
        """
        for n in range(self.planet_count):
            for m in range(self.planet_count):
                if n != self.minimum_interaction_distance:
                    separation = np.subtract(self.pos_array[n], self.pos_array[m])
                    norm_separation = np.linalg.norm(separation)
                    self.acc_array[n] = np.add(self.acc_array[n], -((const.G * self.mass_array[m] * separation) / (norm_separation ** 3)))

    @staticmethod
    @guvectorize(['void(float64[:,:], float64[:], float64[:], int64, float64[:])'],
                 '(b,d),(d),(b),()->(d)', nopython=True, target='parallel')
    def calc_acc_parallel(pos_array, my_pos_array, mass_array, minimum_interaction_distance, acc_out):
        acc_out[0] = 0
        acc_out[1] = 0
        acc_out[2] = 0
        b, d = pos_array.shape
        G = 6.67408e-11
        for m_planet in range(b):
            xsep = my_pos_array[0] - pos_array[m_planet, 0]
            ysep = my_pos_array[1] - pos_array[m_planet, 1]
            zsep = my_pos_array[2] - pos_array[m_planet, 2]
            norm_separation = math.sqrt(xsep * xsep + ysep * ysep + zsep * zsep)
            if norm_separation >= minimum_interaction_distance:
                norm_separation_cubed = norm_separation * norm_separation * norm_separation
                ax = -((G * mass_array[m_planet] * xsep) / norm_separation_cubed)
                ay = -((G * mass_array[m_planet] * ysep) / norm_separation_cubed)
                az = -((G * mass_array[m_planet] * zsep) / norm_separation_cubed)
                acc_out[0] = acc_out[0] + ax
                acc_out[1] = acc_out[1] + ay
                acc_out[2] = acc_out[2] + az

    def get_gpe(self):
        gpe_array = np.zeros((self.planet_count))
        for n in range(self.planet_count):
            gpe = 0
            for m in range(self.planet_count):
                separation = np.subtract(self.pos_array[n], self.pos_array[m])
                norm_separation = np.linalg.norm(separation)
                gpe += -((const.G*self.mass_array[n]*self.mass_array[m])/(1+norm_separation))
            gpe_array[n] = gpe
        return gpe_array

    def check_energy_conservation(self):
        vel_correction = np.zeros((self.planet_count))
        for n in range(self.planet_count):
            gpe = 0
            for m in range(self.planet_count):
                separation = np.subtract(self.pos_array[n], self.pos_array[m])
                norm_separation = np.linalg.norm(separation)
                gpe += ((const.G*self.mass_array[n]*self.mass_array[m])/(1+norm_separation))
            v_mag_current = np.linalg.norm(self.vel_array[n])
            v_mag_max = math.sqrt(abs((self.planets[n].k + gpe) * 2/self.mass_array[n]))
            vel_correction[n] = abs(v_mag_max/v_mag_current)
        return vel_correction

    # def check_energy_conservation_parallel(self, i):
    #     separation = np.linalg.norm(self.pos_array[i])
    #     v_mag_current = np.linalg.norm(self.vel_array[i])
    #     v_mag_max = math.sqrt(abs((self.planets[i].k + ((const.G*self.sun_mass*self.mass_array[i])/(1+separation))) * 2/self.mass_array[i]))
    #     return abs(v_mag_max/v_mag_current)

class Plotting:
    """
    Holds the plot objects
    """
    def __init__(self, simulation):
        self.frames = ((simulation.run_time * 365) // simulation.plot_interval )
        self.lines = []
        self.plots = []
        self.labels = []
        self.fig, (self.ax1) = plt.subplots(nrows=1, ncols=1)
        self.fig.set_size_inches(10, 8)
        self.ax1.set_xlabel("x position (m)")
        self.ax1.set_ylabel("y position (m)")
        self.ax1.set_xlim(-simulation.axes_limit, simulation.axes_limit)
        self.ax1.set_ylim(-simulation.axes_limit, simulation.axes_limit)
        self.timestamp = self.ax1.text(.03, .94, 'Day: ', color='b', transform=self.ax1.transAxes, fontsize='x-large')

    def create_plot(self, simulation):
        for i in range(simulation.planet_count):
            colour = simulation.colours[random.randint(0, 8)]
            if i ==0:
                #sun
                plot = self.ax1.scatter([], [], color="yellow", s=100)
            else:
                #everything else
                plot = self.ax1.scatter([], [], color=colour, s=10)
            label = self.ax1.annotate("", xy=(simulation.planets[i].position[:2]))
            line, = self.ax1.plot([], [], color=colour, linewidth=1)
            self.plots.append(plot)
            self.labels.append(label)
            self.lines.append(line)

    def start_anim(self, simulation):
        ani = animation.FuncAnimation(self.fig, self.animate, repeat=True, frames=self.frames,
                                      fargs=(simulation,), blit=True, interval=0, )
        if simulation.save_anim:
            ani.save('C:/Users/Philip/Pictures/sim/animation2' + str(time.time()) + '.gif', writer='imagemagick', fps=60)
        plt.show()

    def animate(self, i, simulation):
        """
        This is called by the funcAnimation method from matplotlib, which calls it repeatedly to make a live animation.
        Frames is the number of frames in the animation but anim is actually called at least once more at the start of the
        animation. Therefore it is tricky to keep track of the simulation time and so i am keeping time with each planet
        rather than based on the number of frames that have been made.
        """

        current_point = simulation.plot_interval * (3600*24)//simulation.dt * (i + 1) - 1
        start_point = 0
        if current_point - simulation.days_to_plot > 0: start_point = current_point - (simulation.days_to_plot*((3600*24)//(simulation.dt)))
        year = simulation.times[current_point] / (3600 * 24 * 365)

        for n, planet in enumerate(simulation.planets):
            """ 
            Plots the current position and a line plot of its position over the past (days_to_plot) variable
            """
            orbital_radius = planet.orbital_radii[current_point] / 149597870700
            x = np.array([planet.xpoints[current_point], planet.ypoints[current_point]])
            self.plots[n].set_offsets(x)
            self.lines[n].set_xdata(planet.xpoints[start_point:current_point])
            self.lines[n].set_ydata(planet.ypoints[start_point:current_point])
            self.labels[n].set_x(planet.xpoints[current_point])
            self.labels[n].set_y(planet.ypoints[current_point])
            if simulation.plot_labels:
                self.labels[n].set_text("{:.2e}".format(planet.mass/5.972e24) + 'ME' + '\n' + str(round(orbital_radius, 2)) + 'AU')

        self.timestamp.set_text('Year: ' + str(round(year, 4)))
        return self.lines + self.plots + [self.timestamp] + self.labels

    # def delete_plots(self, planets):
    #     deletes planets that have escaped to speed up animation
    #     to_delete = []
    #     for count, planet in enumerate(planets):
    #         if planet.orbital_radius / 149597870700 > 50:
    #             to_delete.append(count)
    #
    #     for count, val in enumerate(to_delete):
    #         planets.pop(val - count)
    #         self.plots.pop(val - count)
    #         self.lines.pop(val - count)
    #         self.labels.pop(val - count)


if __name__ == "__main__":
    """Remove # for profiling"""
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s= io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr,stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
