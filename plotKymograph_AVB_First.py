import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import settings

def plotKymograph_AVB_First(rawdata, r1, AVB_ratio1, AVB_ratio2, FixMaxAVB, fbest):
  rawt = rawdata[:, 1]
  rawsegs = rawdata[:, 2:]

  tall = np.arange(0, len(rawt), 0.05)
  segsall = interp1d(rawt, rawsegs, axis=0)(tall)

  tcut = 10000
  tmax = 30000
  subInds = np.where((tall >= tcut) & (tall <= tmax))

  t = tall[subInds]
  segs = segsall[subInds, :]
  segAmps = np.max(segs, axis=0) - np.min(segs, axis=0)
  segsN = np.zeros((len(segs), 6))

  if np.min(segAmps) < 1e-2:
    for ix in np.arange(0, 6):
      seg = segs[:, ix]
      segN = ((seg - np.quantile(seg, 0.05))/(np.quantile(seg, 0.95) - np.quantile(seg, 0.05)) - 0.5)*2
      segsN[:, ix] = segN

  seg1 = segsN[:, 0]
  seg2 = segsN[:, 1]
  seg3 = segsN[:, 2]
  seg4 = segsN[:, 3]
  seg5 = segsN[:, 4]
  seg6 = segsN[:, 5]

  seg1flag = np.zeros([len(seg1), 1])
  seg2flag = np.zeros([len(seg2), 1])
  seg3flag = np.zeros([len(seg3), 1])
  seg4flag = np.zeros([len(seg4), 1])
  seg5flag = np.zeros([len(seg5), 1])
  seg6flag = np.zeros([len(seg6), 1])

  seg1posInds = np.where(seg1 >= 0)
  seg2posInds = np.where(seg2 >= 0)
  seg3posInds = np.where(seg3 >= 0)
  seg4posInds = np.where(seg4 >= 0)
  seg5posInds = np.where(seg5 >= 0)
  seg6posInds = np.where(seg6 >= 0)

  seg1flag[seg1posInds] = 1
  seg2flag[seg2posInds] = 1
  seg3flag[seg3posInds] = 1
  seg4flag[seg4posInds] = 1
  seg5flag[seg5posInds] = 1
  seg6flag[seg6posInds] = 1

  flagmat = np.concatenate((seg1flag, seg2flag, seg3flag, seg4flag, seg5flag, seg6flag), axis=1)

  flagmat[6, :] = np.zeros([len(t), 1])
  flagmat[:, len(t)+1] = np.zeros([1, 7])

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
