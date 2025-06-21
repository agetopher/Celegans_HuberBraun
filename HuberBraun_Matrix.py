import numpy as np

def HuberBraun_Matrix(t, y, params):
  # Set Global Parmeters
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

  V = y[0:numCells]
  asd = y[numCells:2*numCells]
  asr = y[2*numCells:3*numCells]
  s = y[3*numCells:4*numCells]

  Vj_minus_Vi = np.meshgrid(V,V)

  asd_inf = 1/(1+np.exp(-ssd*(V - V0sd)))

  Isd = rho*gsd*asd*(V-Vsd)
  Isr = rho*gsr*asr*(V-Vsr)
  Il = gl*(V-Vl)

  Isd[PassiveInds] = 0
  Isr[PassiveInds] = 0

  # Connections: Gap Junctions and Synapses
  alpha = 1/(1+np.exp(-beta*(V - vth)))
  IgapMat = GJconn*np.exp(Vj_minus_Vi)
  Igap = ggap*sum(IgapMat, axis=0)

  sMat = np.meshgrid(s).T
  VMat = np.meshgrid(V)

  IsyniMat = Iconn*sMat*(VMat-Esyni)
  Isyni = gsyni*sum(IsyniMat, axis=0).T

  IsyneMat = Iconn*sMat*(VMat-Esyne)
  Isyne = gsyne*sum(IsyneMat, axis=0).T

  IsyniRec = np.append(IsyniRec, np.max(np.max(Isyni)))
  IsyneRec = np.append(IsyneRec, np.max(np.max(Isyne)))

  z = np.zeros(4*numCells)
  z[1:numCells+1] = (Iapp + IAVA + IAVB - Il*NEURONFACTOR2 - Isd*NEURONFACTOR2 - Isr*NEURONFACTOR2 - Igap*NEURONFACTOR2*NEURONFACTOR3 - Isyni*NEURONFACTOR2*NEURONFACTOR3 - Isyne*NEURONFACTOR2*NEURONFACTOR3)/C
  z[numCells+1:2*numCells+1] = (phi/tausd)*(asd_inf - asd)
  z[2*numCells+1:3*numCells+1] = (phi/tausr)*(NEURONFACTOR1*vacc*Isd*NEURONFACTOR2/NEURONFACTOR3 + vdep*asr)
  z[3*numCells+1:4*numCells+1] = ar*alpha*(1-s) - ad*s

  return z