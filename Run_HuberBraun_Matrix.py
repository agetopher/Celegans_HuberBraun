import numpy as np
from scipy.integrate import solve_ivp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Functions import HuberBraun_Matrix
import settings

def Run_HuberBraun_Matrix(type, v, r1, V_prev=0, asd_prev=0, asr_prev=0, s_prev=0):
    settings.IsyniRec = []
    settings.IsyneRec = []

    settings.NEURONFACTOR1 = 70.686
    settings.NEURONFACTOR2 = 100/7.068
    settings.NEURONFACTOR3 = 1000

    r1 = int(r1)

    # Load Classification Data
    OscillatorClassification = []
    with open(f'Data/Comb{r1}_OscillatorsClassification.dat', 'r') as f:
        temp = f.readlines()
        for lin in temp:
            if lin == '\n':
                continue
            OscillatorClassification.append(float(lin.removesuffix('\n')))

    CellsClassification = []
    with open('Data/CellsClassification.dat', 'r') as f:
        temp = f.readlines()
        for lin in temp:
            if lin == '\n':
                continue
            CellsClassification.append(float(lin.removesuffix('\n')))

    # Find Oscillator Indices
    settings.OscInds = []
    for ix in range(len(OscillatorClassification)):
        currOsc = OscillatorClassification[ix]
        for jx in range(len(CellsClassification)):
            currCell = CellsClassification[jx]
            if currOsc == currCell:
                settings.OscInds.append(jx)

    # Set Passive Indices
    settings.PassiveInds = []
    for ix in range(len(CellsClassification)):
        if ix not in settings.OscInds:
            settings.PassiveInds = np.append(settings.PassiveInds, ix)
        settings.PassiveInds = np.array(settings.PassiveInds, dtype=int)

    settings.GJconn = []
    with open('Data/ConnectivityMatrix_SixSegments_GapJunctions.txt', 'r') as f:
        temp = f.readlines()
        for lin in temp:
            if lin == '\n':
                continue
            currLine = lin.removesuffix('\n')
            currLine = currLine.split(',')
            currLine = [int(x) for x in currLine]
            settings.GJconn.append(currLine)
    settings.GJconn = np.array(settings.GJconn, dtype=int)

    settings.Econn = []
    with open('Data/ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt', 'r') as f:
        temp = f.readlines()
        for lin in temp:
            if lin == '\n':
                continue
            currLine = lin.removesuffix('\n')
            currLine = currLine.split(',')
            currLine = [int(x) for x in currLine]
            settings.Econn.append(currLine)
    settings.Econn = np.array(settings.Econn, dtype=int)

    settings.Iconn = []
    with open('Data/ConnectivityMatrix_SixSegments_InhibitorySynapses.txt', 'r') as f:
        temp = f.readlines()
        for lin in temp:
            if lin == '\n':
                continue
            currLine = lin.removesuffix('\n')
            currLine = currLine.split(',')
            currLine = [int(x) for x in currLine]
            settings.Iconn.append(currLine)
    settings.Iconn = np.array(settings.Iconn, dtype=int)

    settings.numCells = 102

    AVAs = [0,1, 2, 9, 10]
    AVBs = [0,1, 3, 7, 8]
    dorsalmuscle = [11, 12, 13]
    ventralmuscle = [14, 15, 16]
    muscle = [11, 12, 13, 14, 15, 16]

    AVAinds = []
    AVBinds = []
    dorsalmuscleInds = []
    ventralmuscleInds = []
    muscleInds = []
    for ix in range(6):
        for jx in range(len(AVAs)):
            AVAinds.append(AVAs[jx] + ix*17)
        for jx in range(len(AVBs)):
            AVBinds.append(AVBs[jx] + ix*17)
        for jx in range(len(dorsalmuscle)):
            dorsalmuscleInds.append(dorsalmuscle[jx] + ix*17)
        for jx in range(len(ventralmuscle)):
            ventralmuscleInds.append(ventralmuscle[jx] + ix*17)
        for jx in range(len(muscle)):
            muscleInds.append(muscle[jx] + ix*17)

    if type == 'AVA':
        AVAstrength = v[0]
        AVBstrength = 0
    elif type == 'AVB':
        AVAstrength = 0
        AVBstrength = v[0]

    settings.IAVA = np.zeros(settings.numCells).reshape(settings.numCells, 1)
    settings.IAVB = np.zeros(settings.numCells).reshape(settings.numCells, 1)

    settings.IAVA[AVAinds] = AVAstrength * settings.NEURONFACTOR2 * 1000
    settings.IAVB[AVBinds] = AVBstrength * settings.NEURONFACTOR2 * 1000

    settings.ggap = v[1]
    settings.gsyne= v[2]
    settings.gsyni = v[3]

    settings.Iapp = 0
    settings.C = 100/7.068
    settings.phi = 0.167
    settings.rho = 0.607

    settings.gl = 0.1
    settings.gsd = 0.25
    settings.gsr = 0.39

    settings.Vl = -60
    settings.Vsd = 50
    settings.Vsr = -90

    settings.tausd = 10
    settings.tausr = 20
    settings.vacc = 0.012
    settings.vdep = 0.17

    settings.V0sd = -40
    settings.ssd = 0.09

    settings.Esyne = 0
    settings.Esyni = -85

    settings.ar = 1
    settings.ad = 5
    settings.beta = 0.125
    settings.vth = -20
    
    t0 = 0
    tf = 30000
    tin = list(range(t0, tf, 10))

    inits_V = []
    with open('Data/InitialVoltages.dat', 'r') as f:
        temp = f.readlines()
        for lin in temp:
            if lin == '\n':
                continue
            currLine = lin.removesuffix('\n')
            currLine = currLine.split(',')
            currLine = [float(x) for x in currLine]
            inits_V.append(currLine)

    if V_prev == 0 and asd_prev == 0 and asr_prev == 0 and s_prev == 0:
        inits_V = np.array(inits_V).reshape(settings.numCells, 1)
        inits_asd = 1/(1+np.exp(-settings.ssd*(inits_V - settings.V0sd))).reshape(settings.numCells, 1)
        inits_asr = ((inits_V + 60)/10.0).reshape(settings.numCells, 1)
        inits_s = np.zeros((settings.numCells, 1))
    else:
        inits_V = V_prev.reshape(settings.numCells, 1)
        inits_asd = asd_prev.reshape(settings.numCells, 1)
        inits_asr = asr_prev.reshape(settings.numCells, 1)
        inits_s = s_prev.reshape(settings.numCells, 1)

    inits = np.concatenate([inits_V, inits_asd, inits_asr, inits_s], axis=0)

    inits = inits.reshape(4*settings.numCells)

    sol = solve_ivp(HuberBraun_Matrix, [t0, tf], inits, method='BDF', t_eval=tin)
    if sol.t.shape[0] < 2000:
        return np.zeros(2000)

    V = sol.y[0:settings.numCells, :]
    asd = sol.y[settings.numCells:2*settings.numCells, :]
    asr = sol.y[2*settings.numCells:3*settings.numCells, :]
    s = sol.y[3*settings.numCells:4*settings.numCells, :]

    muscle_V = V[muscleInds, :]
    dorsalmuscles_V = V[dorsalmuscleInds, :]
    ventralmuscles_V = V[ventralmuscleInds, :]

    dorsal_median_normed = np.divide(dorsalmuscles_V - (np.ones(dorsalmuscles_V.shape) * np.median(dorsalmuscles_V, axis=1).reshape(18, 1)), (1.25 * (np.quantile(dorsalmuscles_V, 0.8, axis=1) - np.quantile(dorsalmuscles_V, 0.2, axis=1))).reshape(18, 1))
    ventral_median_normed = np.divide(ventralmuscles_V - (np.ones(ventralmuscles_V.shape) * np.median(ventralmuscles_V, axis=1).reshape(18, 1)), (1.25 * (np.quantile(ventralmuscles_V, 0.8, axis=1) - np.quantile(ventralmuscles_V, 0.2, axis=1))).reshape(18, 1))

    relative_strengths = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5])

    dorsal_slices = np.array([]).reshape(95, 0)
    ventral_slices = np.array([]).reshape(95, 0)
    for i in range(2000):
        dorsal_slice = np.array([])
        ventral_slice = np.array([])
        for j in range(18):
            curr_dorsal = dorsal_median_normed[j, i+1000] * relative_strengths
            curr_ventral = ventral_median_normed[j, i+1000] * relative_strengths

            if j == 0:
                dorsal_slice = np.append(dorsal_slice, curr_dorsal[0:5], axis=0).reshape(5, 1)
                ventral_slice = np.append(ventral_slice, curr_ventral[0:5], axis=0).reshape(5, 1)
                prev_dorsal = curr_dorsal
                prev_ventral = curr_ventral
                continue

            dorsal_slice = np.append(dorsal_slice, (curr_dorsal[:5] + prev_dorsal[5:]).reshape(5, 1), axis=0)
            ventral_slice = np.append(ventral_slice, (curr_ventral[:5] + prev_ventral[5:]).reshape(5, 1), axis=0)

            if j == 17:
                dorsal_slice = np.append(dorsal_slice, curr_dorsal[5:].reshape(5, 1), axis=0)
                ventral_slice = np.append(ventral_slice, curr_ventral[5:].reshape(5, 1), axis=0)

        dorsal_slices = np.append(dorsal_slices, dorsal_slice, axis=1)
        ventral_slices = np.append(ventral_slices, ventral_slice, axis=1)

    median_normed = dorsal_slices - ventral_slices

    return 8*median_normed