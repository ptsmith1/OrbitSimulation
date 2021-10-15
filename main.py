import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import cProfile, pstats, io
from generate_system import *

def main():
    start = time.time_ns()
    Simulation()
    end = time.time_ns()
    print("Run time: ", (end - start)/1e9, "seconds")

class Simulation:
    """
    Holds the top level methods and parameters
    """
    def __init__(self):
        """Simulation parameters"""
        self.random_planets = True  # if false use our solar system
        self.planet_count = 25  # planets to create, max 9 if using our solar system
        self.days_to_plot = 100  # the number of days for which the position of each planet is plotted
        self.dt = 3600 * 24  # in seconds
        self.run_time = 25  # in years
        self.sun_mass = 1.989e30
        """Plotting parameters"""
        self.plot_data = False  # choose whether to plot the positions of planets
        self.axes_limit = 2000e9  # size of plot window in meters
        self.plot_interval = 25  # in days
        self.delete_distance = 30  # the distance from the Sun at which a planet is deleted in AU
        self.colours = ['gray', 'orange', 'blue', 'chocolate','brown','hotpink','black','green','violet']  # line colours
        """--------------------"""

        if not self.random_planets:
            self.planet_count = 9

        self.planets = []
        self.create_planets()

        if self.plot_data:
            self.plot = Plotting(self)
            self.create_plot()
            self.plot.start_anim(self)

        else:
            for i in range(self.run_time * 365):
                self.evolve(self.planets)
                if i % 25 == 0:
                    print("Year:", self.planets[0].time / (self.dt * 365))
                    print("ID:", self.planets[0].ID, "Position:", self.planets[0].position, "Velocity", self.planets[0].velocity, "Acceleration", self.planets[0].acc)

    def create_planets(self):
        for i in range(self.planet_count):
            """
            Generates the planets with random velocity/positions or based off our solar system
            """
            if self.random_planets:
                self.planets.append(GenerateRandomBody(i + 1))

            else:
                self.planets.append(GenerateStockBody(i+1))

    def create_plot(self):
        self.plot.create_plot(self)

    def evolve(self, planets):
        """
        Updates the location of each planet based on its acceleration due to the Suns gravitational field only
        """
        length = len(planets)
        # acc here stores a list of accelerations for each planet
        acc = self.calc_planet_attraction(planets, length)
        for n, planet in enumerate(planets):
            planet.position += planet.velocity * self.dt
            planet.orbital_radius = np.linalg.norm(planet.position)
            planet.acc = -((const.G * self.sun_mass * planet.position) / (planet.orbital_radius ** 3))
            planet.acc += self.collapse_acc(acc, planet, length, n)
            planet.velocity += planet.acc * self.dt
            planet.xpoints.append(planet.position[0])
            planet.ypoints.append(planet.position[1])
            planet.time += self.dt
            if len(planet.xpoints) >= self.days_to_plot:
                planet.xpoints.pop(0)
                planet.ypoints.pop(0)

    def calc_planet_attraction(self, planets, length):
        """
        Calculates the accelreration between each planet and every other planet, makes sure to not calculate the
        attraction between eg p1 to p2 and between p2 to p1, instead using the first calculated value and changing
        the sign.
        """
        acc = []
        for n, planet_n in enumerate(planets):
            for m, planet_m in enumerate(planets):
                if n == m:
                    acc.append(np.array([0,0,0]))
                elif n>m:
                    acc.append(-acc[m * length + n])
                else:
                    seperation = np.subtract(planet_n.position, planet_m.position)
                    norm_seperation = np.linalg.norm(seperation)
                    acc.append(np.array(-((const.G * planet_n.mass * seperation) / (norm_seperation ** 3))))
        return acc

    def collapse_acc(self, acc, planet, length, n):
        """
        Sums the component's of acc that apply to planet into a single acceleration which is added to the component
        from the sun
        """
        acc_sum = np.array([0.0,0.0,0.0])
        for m in range(length):
            acc_sum += acc[length * n + m]
        return acc_sum

class Plotting:
    """
    Holds the plot objects
    """
    def __init__(self, simulation):
        self.frames = ((simulation.run_time * 365) // simulation.plot_interval) + 1
        self.lines = []
        self.plots = []
        self.labels = []
        self.save_anim = False
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, ncols=1)
        self.ax1.set_xlim(-simulation.axes_limit, simulation.axes_limit)
        self.ax1.set_ylim(-simulation.axes_limit, simulation.axes_limit)
        self.timestamp = self.ax1.text(.03, .94, 'Day: ', color='b', transform=self.ax1.transAxes, fontsize='x-large')
        self.sun = self.ax1.scatter([0], [0], color='yellow', s=100)

        # Time plot variables
        self.times = []
        self.times.append(time.time_ns())
        self.time_steps = []
        self.x_time_data = []
        self.time_plot = self.ax2.plot(self.x_time_data, self.time_steps)

    def create_plot(self, simulation):
        for i in range(simulation.planet_count):
            colour = simulation.colours[random.randint(0, 8)]
            plot = self.ax1.scatter([], [], color=colour, s=10)
            label = self.ax1.annotate(simulation.planets[i].ID, xy=(simulation.planets[i].position[:2]))
            line, = self.ax1.plot([], [], color=colour, linewidth=1)
            self.plots.append(plot)
            self.labels.append(label)
            self.lines.append(line)

    def start_anim(self, simulation):
        ani = animation.FuncAnimation(self.fig, self.animate, repeat=False, frames=self.frames,
                                      fargs=(simulation,), blit=True, interval=0, )
        if self.save_anim:
            ani.save('C:/Users/Philip/Pictures/sim/animation2' + str(time.time()) + '.gif', writer='imagemagick', fps=60)
        plt.show()

    def animate(self, i, simulation):
        """
        This is called by the funcAnimation method from matplotlib, which calls it repeatedly to make a live animation.
        Frames is the number of frames in the animation but anim is actually called at least once more at the start of the
        animation. Therefore it is tricky to keep track of the simulation time and so i am keeping time with each planet
        rather than based on the number of frames that have been made.
        """
        self.delete_plots(simulation.planets)  # deletes old plots
        self.plot_time(i)  # creates the data for, and updates the time plot
        print("dt(frame) = ", self.time_steps[i], " ms")
        if len(simulation.planets) > 0:
           year = simulation.planets[-1].time / (3600 * 24 * 365)
        else:
            plt.close()

        if year >= simulation.run_time:
            # ends the simulation
            plt.close()
        for count in range(simulation.plot_interval):
            simulation.evolve(simulation.planets)

        for n, planet in enumerate(simulation.planets):
            """ 
            Calls the evolve method x times per animation frames where x=plot_interval and then updates all the plot
            items. This is a lot quicker than updating the animation each time each planets position is updated.
            """
            orbital_radius = planet.orbital_radius / 149597870700
            self.plots[n].set_offsets(planet.position[:2])
            self.lines[n].set_xdata(planet.xpoints)
            self.lines[n].set_ydata(planet.ypoints)
            self.labels[n].set_x(planet.position[0])
            self.labels[n].set_y(planet.position[1])
            self.labels[n].set_text(str(round(orbital_radius, 3)) + 'AU')

        self.timestamp.set_text('Year: ' + str(round(year, 4)))
        return self.lines + self.plots + [self.timestamp] + self.labels + self.time_plot

    def plot_time(self, i):
        self.times.append(time.time_ns())
        self.time_steps.append((self.times[i + 1] - self.times[i]) / 1e6)
        self.x_time_data.append(len(self.times) - 1)
        self.time_plot[0].set_xdata(self.x_time_data)
        self.time_plot[0].set_ydata(self.time_steps)
        self.ax2.set_xlim(0, len(self.x_time_data))
        self.ax2.set_ylim(0, max(self.time_steps) + 10)

    def delete_plots(self, planets):
        to_delete = []
        for count, planet in enumerate(planets):
            if planet.orbital_radius / 149597870700 > 50:
                to_delete.append(count)

        for count, val in enumerate(to_delete):
            planets.pop(val - count)
            self.plots.pop(val - count)
            self.lines.pop(val - count)
            self.labels.pop(val - count)


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s= io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr,stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
