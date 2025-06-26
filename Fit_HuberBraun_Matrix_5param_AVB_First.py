import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Functions import get_scores, plotKymograph_AVB_First, HuberBraun_Matrix
import settings

def Fit_HuberBraun_Matrix_5param_AVB_First(v, r1):
  settings.IsyniRec = []
  settings.IsyneRec = []

  settings.NEURONFACTOR1 = 70.686
  settings.NEURONFACTOR2 = 100/7.068
  settings.NEURONFACTOR3 = 1000

  # Load Classification Data
  OscillatorClassification = []
  with open('Data/Comb44_OscillatorsClassification.dat', 'r') as f:
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
      settings.PassiveInds.append(ix)
  settings.PassiveInds = np.array(settings.PassiveInds)

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
  settings.GJconn = np.array(settings.GJconn)

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
  settings.Econn = np.array(settings.Econn)

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
  settings.Iconn = np.array(settings.Iconn)

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
  tin = list(range(t0, tf, 50))

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

  inits_V = np.array(inits_V)
  inits_asd = 1/(1+np.exp(-settings.ssd*(inits_V - settings.V0sd)))
  inits_asr = (inits_V + 60)/10
  inits_s = np.zeros((settings.numCells, 1))

  inits = np.concatenate([inits_V, inits_asd, inits_asr, inits_s], axis=0)

  inits = inits.reshape(4*settings.numCells)

  sol = solve_ivp(HuberBraun_Matrix, [t0, tf], inits, method='BDF', t_eval=tin)

  V = sol.y[0:settings.numCells, :]
  asd = sol.y[settings.numCells:2*settings.numCells, :]
  asr = sol.y[2*settings.numCells:3*settings.numCells, :]
  s = sol.y[3*settings.numCells:4*settings.numCells, :]

  muscle_V = V[muscleInds, :]
  dorsalmuscles_V = V[dorsalmuscleInds, :]

  dorsal_Seg1 = sum(dorsalmuscles_V[0:3, :])
  dorsal_Seg2 = sum(dorsalmuscles_V[3:6, :])
  dorsal_Seg3 = sum(dorsalmuscles_V[6:9, :])
  dorsal_Seg4 = sum(dorsalmuscles_V[9:12, :])
  dorsal_Seg5 = sum(dorsalmuscles_V[12:15, :])
  dorsal_Seg6 = sum(dorsalmuscles_V[15:18, :])

  ventralmuscles_V = V[ventralmuscleInds, :]
  ventral_Seg1 = sum(ventralmuscles_V[0:3, :])
  ventral_Seg2 = sum(ventralmuscles_V[3:6, :])
  ventral_Seg3 = sum(ventralmuscles_V[6:9, :])
  ventral_Seg4 = sum(ventralmuscles_V[9:12, :])
  ventral_Seg5 = sum(ventralmuscles_V[12:15, :])
  ventral_Seg6 = sum(ventralmuscles_V[15:18, :])

  d_minus_v_Seg1 = dorsal_Seg1 - ventral_Seg1
  d_minus_v_Seg2 = dorsal_Seg2 - ventral_Seg2
  d_minus_v_Seg3 = dorsal_Seg3 - ventral_Seg3
  d_minus_v_Seg4 = dorsal_Seg4 - ventral_Seg4
  d_minus_v_Seg5 = dorsal_Seg5 - ventral_Seg5
  d_minus_v_Seg6 = dorsal_Seg6 - ventral_Seg6

  data = np.array([sol.t, d_minus_v_Seg1, d_minus_v_Seg2, d_minus_v_Seg3, d_minus_v_Seg4, d_minus_v_Seg5, d_minus_v_Seg6])
  scores = get_scores(data, 'AVB')
  print(scores)

  SCOAVB = scores[0]
  IncCount = scores[3]
  ratio1AVB = scores[-2]
  DecCount = scores[4]
  ratio2AVB = scores[-1]

  BestMaxAVB = IncCount + DecCount
  V_AVB = V[-1, :].T
  asd_AVB = asd[-1, :].T
  asr_AVB = asr[-1, :].T
  s_AVB = s[-1, :].T
  data_AVB_1 = data;

  H1 = plt.figure(1)
  plotKymograph_AVB_First(data_AVB_1, r1, ratio1AVB, ratio2AVB, BestMaxAVB, SCOAVB)
  plt.savefig(f'Data/Kymograph_Comb{r1}_AVB_First.png', format='png')
  
  return np.concatenate([SCOAVB, ratio1AVB, ratio2AVB, IncCount, DecCount, V_AVB, asd_AVB, asr_AVB, s_AVB, data_AVB_1])
