import numpy as np
from fenics import *

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
)
from simple_worm.worm import Worm

# Parameters
N = 95  # Number of body points - recommend ~100
T = 2.0  # Final time - recommend several undulations
dt = 0.01  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)


def example1():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    u = SpatialCoordinate(worm.V)[0]
    print(f"{u=}")
    control = Function(worm.V)

    # holder for other control directions
    zero = Function(worm.V)
    zero.vector = 0

    # wave parameters
    A = 10.0
    lam = 1.5
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t):
        return A * sin(2.0 * pi * lam * u - 2 * pi * omega * t)

    t = 0.0
    curvature_array = np.zeros((N, int(T/dt)))
    while t < T:
        t += dt

        # update control
        project(alpha_forcing(t), function=control)

        # solve
        ret = worm.update_solution(ControlsFenics(alpha=control, beta=zero, gamma=zero))

        # output variables as 'fenics functions
        # x = ret.x
        # curvature = ret.alpha

        ret_np = ret.to_numpy()
        x_np = ret_np.x
        curvature_np = ret_np.alpha

        # Curvature array
        curvature_array[:, int((t-dt)/dt)] = curvature_np

        print(f"{x_np=}")
        print(f"{curvature_np=}")

    np.savetxt("Data/curvature_array_example1.csv", curvature_array, delimiter=",")


def example2():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """

    params = [44]

    for param in params:

        # Load neuro-muscle data
        neuro_muscle = []
        with open(f"Data/AVB_median_norm_{param}.txt", "r") as f:
            temp = f.readlines()
            for line in temp:
                if line == "\n":
                    continue
                curr_line = line.removesuffix("\n")
                curr_line = curr_line.split(",")
                curr_line = [float(x) for x in curr_line]
                neuro_muscle.append(curr_line)
        neuro_muscle = np.array(neuro_muscle)

        # holders for 'worm', u and control
        N = 95
        dt = 0.05
        worm = Worm(N, dt)
        worm.initialise()

        # controls holder but we will only use 7 different values
        N_controls = 7
        control = np.empty(N)

        # holder for other control directions
        zeroN = np.zeros(N)
        zeroNm = np.zeros(N - 1)

        
        T = neuro_muscle.shape[1] * dt

        t = 0.0
        curvature_array = np.zeros((N, neuro_muscle.shape[1]))
        while t < T:
            t += dt

            # update control | Set this to be a numpy array
            control = neuro_muscle[:, int((t-dt)/dt)].copy().reshape(N)

            # solve
            C = ControlsNumpy(alpha=control, beta=zeroN, gamma=zeroNm)
            ret = worm.update_solution(C.to_fenics(worm))

            # output variables as 'fenics functions
            # x = ret.x
            # curvature = ret.alpha

            ret_np = ret.to_numpy()
            x_np = ret_np.x
            curvature_np = ret_np.alpha

            # Curvature array
            curvature_array[:, int((t-dt)/dt)] = curvature_np

            # print(f"{x_np=}")
            # print(f"{curvature_np=}")

        np.savetxt(f"Data/median_norm_curvature_array_avg_{param}.csv", curvature_array, delimiter=",")


if __name__ == "__main__":
    example2()