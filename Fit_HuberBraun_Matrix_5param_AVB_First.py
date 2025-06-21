import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HuberBraun_Matrix import HuberBraun_Matrix

def Fit_HuberBraun_Matrix_5param_AVB_First(t, y, params):
  # Set Global Parameters
  global Iapp, C, gl, gsd, gsr, Vl, Vsd, Vsr, rho, phi
  global ssd, V0sd, tausd, tausr
  global vacc, vdep
  global ggap, gsyne, gsyni, beta, vth, ar, ad, Esyne, Esyni

  global numCells
  global GJconn, Econn, Iconn
  global IAVA, IAVB 

  global NEURONFACTOR1, NEURONFACTOR2, NEURONFACTOR3

  global IsyniRec, IsyneRec

  global PassiveInds

  IsyniRec = []
  IsyneRec = []

  NEURONFACTOR1 = 70.686
  NEURONFACTOR2 = 100/7.068
  NEURONFACTOR3 = 1000

  with open('data/Comb44_OscillatorsClassification.dat', 'r') as f:
    temp = f.readlines()
    OscillatorClassification = np.array([[float(x) for x in line.split('\n')] for line in temp])
  
  with open('data/CellsClassification.dat', 'r') as f:
    temp = f.readlines()
    CellsClassification = np.array([[float(x) for x in line.split('\n')] for line in temp])
  
  OscInds = []
  for ix in range(len(OscillatorClassification)):
    currOsc = OscillatorClassification[ix,0]

  