import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import settings

def plotKymograph_AVB_First(rawdata, r1, AVB_ratio1, AVB_ratio2, FixMaxAVB, fbest):
  rawt = rawdata[0, :]
  rawsegs = rawdata[1:, :]

  tall = np.arange(rawt[0], rawt[-1], 0.05)
  segsall = interp1d(rawt.T, rawsegs, axis=1)(tall)

  tcut = 10000
  tmax = 30000
  subInds = np.where((tall >= tcut) & (tall <= tmax))

  t = tall[subInds]
  segs = segsall[:, subInds]
  segAmps = np.max(segs, axis=1) - np.min(segs, axis=1)
  segsN = np.zeros(segs.shape)

  if np.min(segAmps) < 1e-2:
    for ix in np.arange(0, 6):
      seg = segs[ix, :]
      segN = ((seg - np.quantile(seg, 0.05))/(np.quantile(seg, 0.95) - np.quantile(seg, 0.05)) - 0.5)*2
      segsN[ix, :] = segN

  seg1 = segsN[0, :]
  seg2 = segsN[1, :]
  seg3 = segsN[2, :]
  seg4 = segsN[3, :]
  seg5 = segsN[4, :]
  seg6 = segsN[5, :]

  seg1flag = np.zeros([1, len(seg1)])
  seg2flag = np.zeros([1, len(seg2)])
  seg3flag = np.zeros([1, len(seg3)])
  seg4flag = np.zeros([1, len(seg4)])
  seg5flag = np.zeros([1, len(seg5)])
  seg6flag = np.zeros([1, len(seg6)])

  seg1posInds = np.where(np.any(seg1 >= 0))
  seg2posInds = np.where(np.any(seg2 >= 0))
  seg3posInds = np.where(np.any(seg3 >= 0))
  seg4posInds = np.where(np.any(seg4 >= 0))
  seg5posInds = np.where(np.any(seg5 >= 0))
  seg6posInds = np.where(np.any(seg6 >= 0))

  seg1flag[seg1posInds] = 1
  seg2flag[seg2posInds] = 1
  seg3flag[seg3posInds] = 1
  seg4flag[seg4posInds] = 1
  seg5flag[seg5posInds] = 1
  seg6flag[seg6posInds] = 1

  flagmat = np.array([seg1flag.T, seg2flag.T, seg3flag.T, seg4flag.T, seg5flag.T, seg6flag.T, np.zeros(seg1flag.shape)])
  print(flagmat.shape)
  flagmat = np.concatenate((flagmat, np.zeros((1, len(t)+1))), axis=0)

  # Make black and white kymograph 
  plt.set(0, 'DefaultAxesFontSize', 24)

  plt.subplot(2, 1, 1)
  plt.plot(t/1000, segsN[:, 0], 'k', linewidth=3)
  plt.plot(t/1000, segsN[:, 1], 'b', linewidth=3)
  plt.plot(t/1000, segsN[:, 2], 'c', linewidth=3)
  plt.plot(t/1000, segsN[:, 3], 'r', linewidth=3)
  plt.plot(t/1000, segsN[:, 4], 'm', linewidth=3)
  plt.plot(t/1000, segsN[:, 5], 'g', linewidth=3)
  plt.plot(t/1000, 0*t, "Color", [0.5, 0.5, 0.5], linewidth=3)
  plt.set(plt.gca, 'box', 'off')
  plt.xlabel('time (s)')
  plt.legend('segment 1', '2', '3', '4', '5', '6', 'Location', 'NorthOutside', 'Orientation', 'Horizontal')
  plt.legend('boxoff')
  plt.xlim([np.min(t/1000), np.max(t/1000)])
  plt.title(f'C%d AVB (R1)%d (R2)%d SQ%d Score%d' % (r1, AVB_ratio1, AVB_ratio2, FixMaxAVB, fbest))

  plt.subplot(2, 1, 2)
  plt.pcolor(flagmat)
  plt.colormap('gray')
  plt.shading('flat')
  plt.set(plt.gca, 'YTick', [1.5, 2.5, 3.5, 4.5, 5.5, 6.6], 'YTickLabel', [6, 5, 4, 3, 2, 1], 'XTick', np.arange(0, len(t), 20000), 'XTickLabel', np.round(t[0:20000:]/1000))
  plt.xlabel('time (s)')
  plt.ylabel('segment')

  return 0
