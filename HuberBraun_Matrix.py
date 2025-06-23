import numpy as np
import settings
import Fit_HuberBraun_Matrix_5param_AVB_First

def HuberBraun_Matrix(t, y, params=None):
  V = y[0:settings.numCells].reshape(settings.numCells, 1)
  asd = y[settings.numCells:2*settings.numCells].reshape(settings.numCells, 1)
  asr = y[2*settings.numCells:3*settings.numCells].reshape(settings.numCells, 1)
  s = y[3*settings.numCells:4*settings.numCells].reshape(settings.numCells, 1)

  Vj_minus_Vi = np.array(np.meshgrid(V)) - np.array(np.meshgrid(V)).T

  asd_inf = 1/(1+np.exp(-settings.ssd*(V - settings.V0sd)))

  Isd = settings.rho*settings.gsd*asd*(V-settings.Vsd)
  Isr = settings.rho*settings.gsr*asr*(V-settings.Vsr)
  Il = settings.gl*(V-settings.Vl)

  Isd[settings.PassiveInds] = 0
  Isr[settings.PassiveInds] = 0

  # Connections: Gap Junctions and Synapses
  alpha = 1/(1+np.exp(-settings.beta*(V - settings.vth)))
  IgapMat = settings.GJconn * Vj_minus_Vi
  Igap = settings.ggap * np.sum(IgapMat, axis=1).T
  Igap = Igap.reshape(settings.numCells, 1)

  sMat, sloss = np.meshgrid(s, s)
  VMat, Vloss = np.meshgrid(V, V)
  sMat = sMat.T

  IsyniMat = settings.Iconn.T*sMat*(np.subtract(VMat, settings.Esyni))
  Isyni = settings.gsyni * np.sum(IsyniMat, axis=1).T
  Isyni = Isyni.reshape(settings.numCells, 1)

  IsyneMat = settings.Econn.T*sMat*(np.subtract(VMat, settings.Esyne))
  Isyne = settings.gsyne * np.sum(IsyneMat, axis=1).T
  Isyne = Isyne.reshape(settings.numCells, 1)

  settings.IsyniRec = np.append(settings.IsyniRec, np.max(np.max(Isyni)))
  settings.IsyneRec = np.append(settings.IsyneRec, np.max(np.max(Isyne)))

  z = np.concatenate([V, asd, asr, s]).reshape(4*settings.numCells,)
  z[0:settings.numCells] = ((settings.Iapp + settings.IAVA + settings.IAVB - Il*settings.NEURONFACTOR2 - Isd*settings.NEURONFACTOR2 - Isr*settings.NEURONFACTOR2 - Igap*settings.NEURONFACTOR2*settings.NEURONFACTOR3 - Isyni*settings.NEURONFACTOR2*settings.NEURONFACTOR3 - Isyne*settings.NEURONFACTOR2*settings.NEURONFACTOR3)/settings.C).reshape(settings.numCells,)
  z[settings.numCells:2*settings.numCells] = (settings.phi/settings.tausd)*(asd_inf - asd).reshape(settings.numCells,)
  z[2*settings.numCells:3*settings.numCells] = (settings.phi/settings.tausr)*(settings.NEURONFACTOR1*settings.vacc*Isd*settings.NEURONFACTOR2/settings.NEURONFACTOR3 + settings.vdep*asr).reshape(settings.numCells,)
  z[3*settings.numCells:4*settings.numCells] = (settings.ar*alpha*(1-s) - settings.ad*s).reshape(settings.numCells,)

  return z