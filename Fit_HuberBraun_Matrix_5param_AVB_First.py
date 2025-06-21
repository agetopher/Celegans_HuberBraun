import numpy as np
from scipy.integrate import solve_ivp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HuberBraun_Matrix import HuberBraun_Matrix
import settings

def Fit_HuberBraun_Matrix_5param_AVB_First(v, r1):
  settings.IsyniRec = []
  settings.IsyneRec = []

  settings.NEURONFACTOR1 = 70.686
  settings.NEURONFACTOR2 = 100/7.068
  settings.NEURONFACTOR3 = 1000

  # Load Classification Data
  OscillatorClassification = []
  with open('data/Comb44_OscillatorsClassification.dat', 'r') as f:
    temp = f.readlines()
    for lin in temp:
      if lin == '\n':
        continue
      OscillatorClassification.append(float(lin.removesuffix('\n')))

  CellsClassification = []
  with open('data/CellsClassification.dat', 'r') as f:
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
  with open('data/ConnectivityMatrix_SixSegments_GapJunctions.txt', 'r') as f:
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
  with open('data/ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt', 'r') as f:
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
  with open('data/ConnectivityMatrix_SixSegments_InhibitorySynapses.txt', 'r') as f:
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

  AVAs = [1,2, 3, 10, 11]
  AVBs = [1,2, 4, 8, 9]
  dorsalmuscle = [12, 13, 14]
  ventralmuscle = [15, 16, 17]
  muscle = [12, 13, 14, 15, 16, 17]

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
  settings.vdep = 0.017

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
  with open('data/InitialVoltages.dat', 'r') as f:
    temp = f.readlines()
    for lin in temp:
      if lin == '\n':
        continue
      currLine = lin.removesuffix('\n')
      currLine = currLine.split(',')
      currLine = [float(x) for x in currLine]
      inits_V.append(currLine)

  inits_asd = 1/(1+np.exp(-settings.ssd*np.subtract(inits_V, settings.V0sd))) 
  inits_asr = np.divide(np.add(inits_V, 60), 10)
  inits_s = np.zeros(settings.numCells)

  inits_V = np.array(inits_V)
  inits_asd = np.array(inits_asd)
  inits_asr = np.array(inits_asr)
  inits_s = np.array(inits_s).reshape(settings.numCells, 1)

  inits = np.concatenate([inits_V, inits_asd, inits_asr, inits_s], axis=0)

  inits = inits.reshape(4*settings.numCells)

  sol = solve_ivp(HuberBraun_Matrix, [t0, tf], inits, method='BDF', t_eval=tin)

  return sol
