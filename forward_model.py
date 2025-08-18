import numpy as np
from fenics import *

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
)
from simple_worm.worm import Worm
from Fit_HuberBraun_Matrix_5param_AVB_First import *
from Run_HuberBraun_Matrix import *

# Parameters
N = 95  # Number of body points - recommend ~100
T = 20.0  # Final time - recommend several undulations
dt = 0.01  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)


def example1():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
    # wave parameters
    best_parallel = np.zeros(N)
    best_perpendicular = np.zeros(N)
    best_dist = 0
    dists = []

    for i in [8]:
        # holders for 'worm', u and control
        worm = Worm(N, dt)
        worm.initialise()

        u = SpatialCoordinate(worm.V)[0]
        print(f"{u=}")
        control = Function(worm.V)

        # holder for other control directions
        zero = Function(worm.V)
        zero.vector = 0
        
        A = 8
        lam = 1.5
        omega = 1.0

        # specific forcing function
        def alpha_forcing(t):
            return A * sin(2.0 * pi * lam * u - 2 * pi * omega * t)

        t = 0.0
        parallel_com = np.zeros((int(T/dt)))
        perpendicular_com = np.zeros((int(T/dt)))
        while t < T:
            t += dt

            noise = np.random.normal(0, 0.1)
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

            # Store the center of mass data
            parallel_com[int((t-dt)/dt)] = np.sum(x_np[0, :], axis=0) / N
            perpendicular_com[int((t-dt)/dt)] = np.sum(x_np[2, :], axis=0) / N
        
        curr_dist = np.sqrt(np.sum((parallel_com[-1] - parallel_com[0])**2 + (perpendicular_com[-1] - perpendicular_com[0])**2))
        dists.append(curr_dist)
            
        if best_dist < curr_dist:
            best_parallel = parallel_com.copy()
            best_perpendicular = perpendicular_com.copy()
            best_dist = curr_dist

        del worm, u, control, zero

    print(dists)

    np.savetxt("Data/parallel_array_example1.csv", best_parallel, delimiter=",")
    np.savetxt("Data/perpendicular_array_example1.csv", best_perpendicular, delimiter=",")



def example2():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """

    params = [98]

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
        parallel_array = np.zeros((N, neuro_muscle.shape[1]))
        perpendicular_array = np.zeros((N, neuro_muscle.shape[1]))
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
            parallel_array[:, int((t-dt)/dt)] = x_np[0, :]
            perpendicular_array[:, int((t-dt)/dt)] = x_np[2, :]

            # print(f"{x_np=}")
            # print(f"{curvature_np=}")

        np.savetxt(f"Data/mean_norm_curvature_array_avg_{param}.csv", curvature_array, delimiter=",")
        np.savetxt(f"Data/mean_norm_parallel_array_avg_{param}.csv", parallel_array, delimiter=",")
        np.savetxt(f"Data/mean_norm_perpendicular_array_avg_{param}.csv", perpendicular_array, delimiter=",")



def test_model(params):
    N = 95
    dt = 0.01
    T = 20.0
    
    best_parallel = np.zeros(int(T/dt))
    best_perpendicular = np.zeros(int(T/dt))
    best_dist = 0
    dists = []

    k = 1
    last = np.zeros((4, 1))
    for param in params:
        r1 = 81
        v = param
        if np.sum(v == last) == 4:
            k += 1 
            continue
        last = v.copy()
        
        neuro_muscle = Fit_HuberBraun_Matrix_5param_AVB_First(v, r1)

        worm = Worm(N, dt)
        worm.initialise()

        # controls holder but we will only use 7 different values
        N_controls = 7
        control = np.empty(N)

        # holder for other control directions
        zeroN = np.zeros(N)
        zeroNm = np.zeros(N - 1)

        t = 0.0
        # curvature_array = np.zeros((N, neuro_muscle.shape[1]))
        parallel_com = np.zeros((int(T/dt)))
        perpendicular_com = np.zeros((int(T/dt)))
        max_dist = 0
        while t < T:
            t += dt

            # update control | Set this to be a numpy array
            control = neuro_muscle[:, int((t-dt)/dt)].copy().reshape(N)

            # solve
            C = ControlsNumpy(alpha=control, beta=zeroN, gamma=zeroNm)
            ret = worm.update_solution(C.to_fenics(worm))

            ret_np = ret.to_numpy()
            x_np = ret_np.x
            curvature_np = ret_np.alpha

            if np.sum(x_np[0, :]) == 0 and np.sum(x_np[2, :]) == 0:
                print(x_np[0, :])
                print(x_np[2, :])

            # Curvature array
            parallel_com[int((t-dt)/dt)] = np.sum(x_np[0, :], axis=0) / N
            perpendicular_com[int((t-dt)/dt)] = np.sum(x_np[2, :], axis=0) / N


            curr_dist = np.sqrt(np.sum((parallel_com[int((t-dt)/dt)] - parallel_com[0])**2 + (perpendicular_com[int((t-dt)/dt)] - perpendicular_com[0])**2))
            if max_dist < curr_dist:
                max_dist = curr_dist

        dists.append(max_dist)
            
        if best_dist < max_dist:
            best_parallel = parallel_com.copy()
            best_perpendicular = perpendicular_com.copy()
            best_dist = max_dist

        np.savetxt(f"Data/{r1}_{k}_median_norm_parallel.csv", parallel_com, delimiter=",")
        np.savetxt(f"Data/{r1}_{k}_median_norm_perpendicular.csv", perpendicular_com, delimiter=",")

        k += 1

        del worm, control, zeroN, zeroNm

    np.savetxt(f"Data/best_{r1}_median_norm_parallel.csv", best_parallel, delimiter=",")
    np.savetxt(f"Data/best_{r1}_median_norm_perpendicular.csv", best_perpendicular, delimiter=",")

    return best_parallel, best_perpendicular, best_dist, dists

def mechanical_model(neuro_muscle, N, dt, T):
    worm = Worm(N, dt)
    worm.initialise()

    # controls holder but we will only use 7 different values
    control = np.empty(N)

    # holder for other control directions
    zeroN = np.zeros(N)
    zeroNm = np.zeros(N - 1)

    t = 0.0
    # curvature_array = np.zeros((N, neuro_muscle.shape[1]))
    parallel_com = np.zeros((int(T/dt)))
    perpendicular_com = np.zeros((int(T/dt)))
    max_dist = 0
    while t < T:
        t += dt

        # update control | Set this to be a numpy array
        control = neuro_muscle[:, int((t-dt)/dt)].copy().reshape(N)

        # solve
        C = ControlsNumpy(alpha=control, beta=zeroN, gamma=zeroNm)
        ret = worm.update_solution(C.to_fenics(worm))

        ret_np = ret.to_numpy()
        x_np = ret_np.x
        curvature_np = ret_np.alpha

        if np.sum(x_np[0, :]) == 0 and np.sum(x_np[2, :]) == 0:
            print(x_np[0, :])
            print(x_np[2, :])

        # Curvature array
        parallel_com[int((t-dt)/dt)] = np.sum(x_np[0, :], axis=0) / N
        perpendicular_com[int((t-dt)/dt)] = np.sum(x_np[2, :], axis=0) / N 

        curr_dist = np.sqrt(np.sum((parallel_com[int((t-dt)/dt)] - parallel_com[0])**2 + (perpendicular_com[int((t-dt)/dt)] - perpendicular_com[0])**2))
        if max_dist < curr_dist:
            max_dist = curr_dist

    del worm, control, zeroN, zeroNm

    return parallel_com, perpendicular_com, max_dist

"""TODO: Add optimization function"""
def fitness_function(v, r1, N, dt, T, type="AVB", V_prev=0, asd_prev=0, asr_prev=0, s_prev=0):
    neuro_muscle = Run_HuberBraun_Matrix(type, v, r1, V_prev, asd_prev, asr_prev, s_prev)

    if np.sum(neuro_muscle) == 0:
        return 0

    parallel_com, perpendicular_com, max_dist = mechanical_model(neuro_muscle, N, dt, T)

    return max_dist


def optimization_loop(osc_params, N, dt, T): 
    best_osc_params = np.zeros((4, len(osc_params)))
    osc_idx = 0
    for osc_param in osc_params:
        if osc_params.shape[1] > 1:
            r1 = osc_param[0]
            v = osc_param[1:]
        else:
            r1 = osc_param

        NP = 4
        F = 0.5
        CR = 0.2
        max_iter = 20
        D = 4 
        lower_bound = np.array([1e-5, 1e-7, 1e-5, 1e-5])
        upper_bound = np.array([1e-2, 1e-4, 1e-2, 1e-2])
        population = np.zeros((D, NP))

        best_generations = np.zeros((4, max_iter))
        best_fitness = np.zeros(max_iter)

        # Initialize Population
        population[:, 0] = v
        for i in range(1, NP):
            population[:, i] = lower_bound + np.random.uniform(lower_bound, upper_bound)*(upper_bound - lower_bound)
        xbest = 0
        fbest = fitness_function(v, r1, N, dt, T)
        for i in range(1, NP):
            curr_fitness = fitness_function(population[:, i], r1, N, dt, T)
            if curr_fitness > fbest: 
                xbest = i
                fbest = curr_fitness
        # Mutation
        iter = 1
        while iter < max_iter:
            print(f"Oscillator: {osc_param} Iteration: {iter}")
            for i in range(NP):
                # Generate Mutation Vectors
                r = np.zeros((3, 1), dtype=int)
                r[0] = np.floor(np.random.rand()*NP)
                while r[0] == i:
                    r[0] = np.floor(np.random.rand()*NP)
                r[1] = np.floor(np.random.rand()*NP)
                while r[1] == i or r[1] == r[0]:
                    r[1] = np.floor(np.random.rand()*NP)
                r[2] = np.floor(np.random.rand()*NP)
                while r[2] == i or r[2] == r[0] or r[2] == r[1]:
                    r[2] = np.floor(np.random.rand()*NP)

                v = population[:, r[0]] + F*(population[:, r[1]] - population[:, r[2]])
                u = np.zeros((D, 1))
                x = population[:, i]

                # Crossover
                for k in range(D):
                    if np.random.rand() < CR:
                        u[k] = v[k]
                    else:
                        u[k] = x[k]

                # Selection
                u_fitness = fitness_function(u, r1, N, dt, T)
                x_fitness = fitness_function(x, r1, N, dt, T)
                if u_fitness > x_fitness:
                    population[:, i] = u.reshape(D)
                if u_fitness > fbest:
                    xbest = i
                    fbest = u_fitness
            best_generations[:, iter] = population[:, xbest].copy()
            best_fitness[iter] = fbest
            iter += 1 

        best_osc_params[:, osc_idx] = population[:, xbest]
        osc_idx += 1

    return best_osc_params, best_generations, best_fitness, xbest

                
                
                
if __name__ == "__main__":
    paramsAVB = np.array([[44, 0.000363000000000000, 5.45726102941647e-06, 0.00167208415554239, 0.000660000000000000],
                [68, 0.000966628944244780, 8.33918539323647e-06, 0.00208400000000000, 0.00316478079207975],
                [81, 0.000757091833987073, 5.79833984374115e-06, 0.000936577073528412, 0.000512000000000000], 
                [94, 0.000892000000000000, 6.13378099174940e-06, 0.00201196569493983, 0.00502294791546334],
                [98, 0.000889000000000000, 1.11084610156032e-05, 0.00165239584154480, 0.000512000000000000],
                [106, 0.00100200000000000, 1.82358292522831e-05, 0.000903535078875878, 0.000512000000000000],
                [112, 0.00111725138757093, 1.16292177635975e-05, 0.00176500000000000, 0.00191920134428983]])
    paramsAVA = np.array([[44, 0.00138300000000000, 5.45726102941647e-06, 0.00167208415554239, 0.000660000000000000],
                [68, 0.000715663214366652, 8.33918539323647e-06, 0.00208400000000000, 0.00316478079207975],
                [81, 0.00149599294308000, 5.79833984374115e-06, 0.000936577073528412, 0.000512000000000000],
                [94, 0.000537000000000000, 6.13378099174940e-06, 0.00201196569493983, 0.00502294791546334], 
                [98, 0.00104700000000000, 1.11084610156032e-05, 0.00165239584154480, 0.000512000000000000], 
                [106, 0.000675000000000000, 1.82358292522831e-05, 0.000903535078875878, 0.000512000000000000], 
                [112, 0.00126612633536624, 1.16292177635975e-05, 0.00176500000000000, 0.00191920134428983]])
    optim_osc = np.array([[44], [68], [81], [94], [98], [106], [112]])

    # with open("Data/best_generations.csv", "r") as f:
    #     temp = f.readlines()
    #     best_generations = np.zeros((4, 20))
    #     i = 0
    #     for line in temp:
    #         line = line.removesuffix("\n")
    #         line = line.split(",")
    #         line = [float(x) for x in line]
    #         best_generations[i, :] = line
    #         i += 1

    # best_generations = best_generations.T

    # test_model(best_generations[1:, :])



    # example1()
    # test_model()

    best_osc_params, best_generations, best_fitness, best = optimization_loop(paramsAVB, 95, 0.01, 20.0)

    np.savetxt("Data/best_osc_params.csv", best_osc_params, delimiter=",")
    np.savetxt("Data/best_generations.csv", best_generations, delimiter=",")
    np.savetxt("Data/best_fitness.csv", best_fitness, delimiter=",")
    print(best)
