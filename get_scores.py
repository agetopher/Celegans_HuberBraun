import numpy as np
import scipy.interpolate as interp1d

def get_scores(rawdata, input):
  if len(rawdata) > 0:
    if len(rawdata) > 200:
      shortDatFiles = [shortDatFile, filex]
    if np.sum(np.isnan(rawdata)) == 0:
      rawt = rawdata[:, 1]
      rawsegs = rawdata[:, 2:]
      tall = np.arange(rawt[1], rawt[-1], 0.1)
      segsall = interp1d(rawt, rawsegs, axis=0)(tall)
      tcut = 10000
      subInds = tall[tall >= tcut]
      t = tall[subInds]
      segs = segsall[subInds, :]
      segAmps = np.max(segs, axis=0) - np.min(segs, axis=0)
      ftobIncrec = []
      ftobDecrec = []
      btofIncrec = []
      btofDecrec = []
      if np.min(segAmps) > 1e-2:
        segsN = np.zeros((len(segs), 6))
        for ix in range(1, 6):
          seg = segs[ix, :]
          segN = ((seg - np.quantile(seg, 0.5))/(np.quantile(seg, 0.95) - np.quantile(seg, 0.05)) - 0.5)*2
          segsN[:, ix] = segN
        seg1 = segsN[:, 1]
        seg2 = segsN[:, 2]
        seg3 = segsN[:, 3]
        seg4 = segsN[:, 4]
        seg5 = segsN[:, 5]
        seg6 = segsN[:, 6]
        seg1crossIncTimes = np.array([])
        seg2crossIncTimes = np.array([])
        seg3crossIncTimes = np.array([])
        seg4crossIncTimes = np.array([])
        seg5crossIncTimes = np.array([])
        seg6crossIncTimes = np.array([])
        seg1crossDecTimes = np.array([])
        seg2crossDecTimes = np.array([])
        seg3crossDecTimes = np.array([])
        seg4crossDecTimes = np.array([])
        seg5crossDecTimes = np.array([])
        seg6crossDecTimes = np.array([])
        seg1crossIncLabels = np.array([])
        seg2crossIncLabels = np.array([])
        seg3crossIncLabels = np.array([])
        seg4crossIncLabels = np.array([])
        seg5crossIncLabels = np.array([])
        seg6crossIncLabels = np.array([])
        seg1crossDecLabels = np.array([])
        seg2crossDecLabels = np.array([])
        seg3crossDecLabels = np.array([])
        seg4crossDecLabels = np.array([])
        seg5crossDecLabels = np.array([])
        seg6crossDecLabels = np.array([])

        for ix in np.arange(1, len(t)):
          if seg1[ix] > 0 and seg1[ix-1] < 0:
            seg1crossIncTimes = np.append(seg1crossIncTimes, t[ix])
            seg1crossIncLabels = np.append(seg1crossIncLabels, 1)
          if seg2[ix] > 0 and seg2[ix-1] < 0:
            seg2crossIncTimes = np.append(seg2crossIncTimes, t[ix])
            seg2crossIncLabels = np.append(seg2crossIncLabels, 2)
          if seg3[ix] > 0 and seg3[ix-1] < 0:
            seg3crossIncTimes = np.append(seg3crossIncTimes, t[ix])
            seg3crossIncLabels = np.append(seg3crossIncLabels, 3)
          if seg4[ix] > 0 and seg4[ix-1] < 0:
            seg4crossIncTimes = np.append(seg4crossIncTimes, t[ix])
            seg4crossIncLabels = np.append(seg4crossIncLabels, 4)
          if seg5[ix] > 0 and seg5[ix-1] < 0:
            seg5crossIncTimes = np.append(seg5crossIncTimes, t[ix])
            seg5crossIncLabels = np.append(seg5crossIncLabels, 5)
          if seg6[ix] > 0 and seg6[ix-1] < 0:
            seg6crossIncTimes = np.append(seg6crossIncTimes, t[ix])
            seg6crossIncLabels = np.append(seg6crossIncLabels, 6)
          if seg1[ix] < 0 and seg1[ix-1] > 0:
            seg1crossDecTimes = np.append(seg1crossDecTimes, t[ix])
            seg1crossDecLabels = np.append(seg1crossDecLabels, 1)
          if seg2[ix] < 0 and seg2[ix-1] > 0:
            seg2crossDecTimes = np.append(seg2crossDecTimes, t[ix])
            seg2crossDecLabels = np.append(seg2crossDecLabels, 2)
          if seg3[ix] < 0 and seg3[ix-1] > 0:
            seg3crossDecTimes = np.append(seg3crossDecTimes, t[ix])
            seg3crossDecLabels = np.append(seg3crossDecLabels, 3)
          if seg4[ix] < 0 and seg4[ix-1] > 0:
            seg4crossDecTimes = np.append(seg4crossDecTimes, t[ix])
            seg4crossDecLabels = np.append(seg4crossDecLabels, 4)
          if seg5[ix] < 0 and seg5[ix-1] > 0:
            seg5crossDecTimes = np.append(seg5crossDecTimes, t[ix])
            seg5crossDecLabels = np.append(seg5crossDecLabels, 5)
          if seg6[ix] < 0 and seg6[ix-1] > 0:
            seg6crossDecTimes = np.append(seg6crossDecTimes, t[ix])
            seg6crossDecLabels = np.append(seg6crossDecLabels, 6)
        
        allCrossIncLabels = np.concatenate((seg1crossIncLabels, seg2crossIncLabels, seg3crossIncLabels, seg4crossIncLabels, seg5crossIncLabels, seg6crossIncLabels))
        allCrossIncTimes = np.concatenate((seg1crossIncTimes, seg2crossIncTimes, seg3crossIncTimes, seg4crossIncTimes, seg5crossIncTimes, seg6crossIncTimes))
        allCrossIncSlopes = np.ones((len(allCrossIncTimes), 1))
        allCrossDecLabels = np.concatenate((seg1crossDecLabels, seg2crossDecLabels, seg3crossDecLabels, seg4crossDecLabels, seg5crossDecLabels, seg6crossDecLabels))
        allCrossDecTimes = np.concatenate((seg1crossDecTimes, seg2crossDecTimes, seg3crossDecTimes, seg4crossDecTimes, seg5crossDecTimes, seg6crossDecTimes))
        allCrossDecSlopes = np.ones((len(allCrossDecTimes), 1))

      else:
        print('amplitute below threshold')
    else:
      print(".dat file has nan value")
      allCrossSorted = np.nan

  data = allCrossSorted 
  if np.sum(np.sum(np.isnan(data))) == 0:
    times = data[:, 0]
    labels = data[:, 1]
    slopes = data[:, 2]
    # Get Segment Crossings
    seg1IncInds = np.where(labels == 1 and slopes == 1)
    seg2IncInds = np.where(labels == 2 and slopes == 1)
    seg3IncInds = np.where(labels == 3 and slopes == 1)
    seg4IncInds = np.where(labels == 4 and slopes == 1)
    seg5IncInds = np.where(labels == 5 and slopes == 1)
    seg6IncInds = np.where(labels == 6 and slopes == 1)
    seg1DecInds = np.where(labels == 1 and slopes == -1)
    seg2DecInds = np.where(labels == 2 and slopes == -1)
    seg3DecInds = np.where(labels == 3 and slopes == -1)
    seg4DecInds = np.where(labels == 4 and slopes == -1)
    seg5DecInds = np.where(labels == 5 and slopes == -1)
    seg6DecInds = np.where(labels == 6 and slopes == -1)

    # Get Segment Crossing Times
    seg1IncTimes = times[seg1IncInds]
    seg2IncTimes = times[seg2IncInds]
    seg3IncTimes = times[seg3IncInds]
    seg4IncTimes = times[seg4IncInds]
    seg5IncTimes = times[seg5IncInds]
    seg6IncTimes = times[seg6IncInds]
    seg1DecTimes = times[seg1DecInds]
    seg2DecTimes = times[seg2DecInds]
    seg3DecTimes = times[seg3DecInds]
    seg4DecTimes = times[seg4DecInds]
    seg5DecTimes = times[seg5DecInds]
    seg6DecTimes = times[seg6DecInds]

    # Get Alternation Periods
    ftobIncAltPeriods = np.diff(seg1IncTimes)
    ftobDecAltPeriods = np.diff(seg1DecTimes)
    btofIncAltPeriods = np.diff(seg2IncTimes)
    btofDecAltPeriods = np.diff(seg2DecTimes)
    allForwardPhaseDiffs = np.array([])
    allBackwardPhaseDiffs = np.array([])
    allForwardTimeDiffs = np.array([])
    allBackwardTimeDiffs = np.array([])
    ftobIncPropPeriods = np.array([])
    ftobDecPropPeriods = np.array([])
    btofIncPropPeriods = np.array([])
    btofDecPropPeriods = np.array([])
    ftobIncCheckAltPeriods = np.array([])
    ftobDecCheckAltPeriods = np.array([])
    btofIncCheckAltPeriods = np.array([])
    btofDecCheckAltPeriods = np.array([])
    ftobIncCount = 0
    ftobDecCount = 0
    btofIncCount = 0
    btofDecCount = 0
    ftobIncRec = np.zeros((len(seg1IncInds), 6))
    ftobDecRec = np.zeros((len(seg1DecInds), 6))
    btofIncRec = np.zeros((len(seg2IncInds), 6))
    btofDecRec = np.zeros((len(seg2DecInds), 6))
    ftobInc_percet_overlap = np.zeros((len(seg1IncInds), 6))
    ftobDec_percet_overlap = np.zeros((len(seg1DecInds), 6))
    btofInc_percet_overlap = np.zeros((len(seg2IncInds), 6))
    btofDec_percet_overlap = np.zeros((len(seg2DecInds), 6))
    ftobIncPhaseDiffs = np.zeros((len(seg1IncInds), 6))
    ftobDecPhaseDiffs = np.zeros((len(seg1DecInds), 6))
    btofIncPhaseDiffs = np.zeros((len(seg2IncInds), 6))
    btofDecPhaseDiffs = np.zeros((len(seg2DecInds), 6))

    # ftob inc
    for ix in np.arange(0, len(seg1IncInds)):
      percent_overlap = np.array([])
      ftobIncTimes = np.array([])
      ftobIncTimes[0] = seg1IncTimes(ix)
      next_ftobDec_seg1 = np.min(seg1DecTimes[seg1DecTimes > ftobIncTimes[0]])
      next_ftobInc_seg2 = np.min(seg2IncTimes[seg2IncTimes > ftobIncTimes[0]])
      if len(next_ftobDec_seg1) > 0 and len(next_ftobInc_seg2) > 0 and next_ftobInc_seg2 < next_ftobDec_seg1:
        duration = next_ftobDec_seg1 - ftobIncTimes[0]
        overlap = next_ftobDec_seg1 - next_ftobInc_seg2
        percent_overlap[0] = overlap/duration

        ftobIncTimes[1] = next_ftobInc_seg2
        next_ftobDec_seg2 = np.min(seg2DecTimes[seg2DecTimes > ftobIncTimes[1]])
        next_ftobInc_seg3 = np.min(seg3IncTimes[seg3IncTimes > ftobIncTimes[1]])
        if len(next_ftobDec_seg2) > 0 and len(next_ftobInc_seg3) > 0 and next_ftobInc_seg3 < next_ftobDec_seg2:
          duration = next_ftobDec_seg2 - ftobIncTimes[1]
          overlap = next_ftobDec_seg2 - next_ftobInc_seg3
          percent_overlap[1] = overlap/duration

          ftobIncTimes[2] = next_ftobInc_seg3
          next_ftobDec_seg3 = np.min(seg3DecTimes[seg3DecTimes > ftobIncTimes[2]])
          next_ftobInc_seg4 = np.min(seg4IncTimes[seg4IncTimes > ftobIncTimes[2]])
          if len(next_ftobDec_seg3) > 0 and len(next_ftobInc_seg4) > 0 and next_ftobInc_seg4 < next_ftobDec_seg3:
            duration = next_ftobDec_seg3 - ftobIncTimes[2]
            overlap = next_ftobDec_seg3 - next_ftobInc_seg4
            percent_overlap[2] = overlap/duration

            ftobIncTimes[3] = next_ftobInc_seg4
            next_ftobDec_seg4 = np.min(seg4DecTimes[seg4DecTimes > ftobIncTimes[3]])
            next_ftobInc_seg5 = np.min(seg5IncTimes[seg5IncTimes > ftobIncTimes[3]])
            if len(next_ftobDec_seg4) > 0 and len(next_ftobInc_seg5) > 0 and next_ftobInc_seg5 < next_ftobDec_seg4:
              duration = next_ftobDec_seg4 - ftobIncTimes[3]
              overlap = next_ftobDec_seg4 - next_ftobInc_seg5
              percent_overlap[3] = overlap/duration

              ftobIncTimes[4] = next_ftobInc_seg5
              next_ftobDec_seg5 = np.min(seg5DecTimes[seg5DecTimes > ftobIncTimes[4]])
              next_ftobInc_seg6 = np.min(seg6IncTimes[seg6IncTimes > ftobIncTimes[4]])
              if len(next_ftobDec_seg5) > 0 and len(next_ftobInc_seg6) > 0 and next_ftobInc_seg6 < next_ftobDec_seg5:
                duration = next_ftobDec_seg5 - ftobIncTimes[4]
                overlap = next_ftobDec_seg5 - next_ftobInc_seg6
                percent_overlap[4] = overlap/duration

                ftobIncTimes[5] = next_ftobInc_seg6

      checkAltPeriod = seg1IncTimes[ix+1]
      if len(ftobIncTimes) == 6:
        ftobIncCount = ftobIncCount + 1
        ftobIncPropPeriod = ftobIncTimes[5] - ftobIncTimes[0]
        ftobIncPropPeriods = np.append(ftobIncPropPeriods, ftobIncPropPeriod)
        ftobIncCheckAltPeriods = np.append(ftobIncCheckAltPeriods, checkAltPeriod - ftobIncTimes[ix])
        ftobIncRec[:, ix] = ftobIncTimes.T
        ftobInc_percet_overlap[:, ix] = percent_overlap.T
        ftobIncTimesDiff = np.diff(ftobIncTimes)
        ftobIncPhaseDiffs[:, ix] = ftobIncTimesDiff/ftobIncPropPeriod
        allForwardPhaseDiffs = np.append(allForwardPhaseDiffs, ftobIncPhaseDiffs[:, ix])
        allForwardTimeDiffs = np.append(allForwardTimeDiffs, ftobIncTimesDiff.T)
    
    # ftob dec
    for ix in np.arange(0, len(seg1DecInds)):
      percent_overlap = np.array([])
      ftobDecTimes = np.array([])
      ftobDecTimes[0] = seg1DecTimes(ix)
      next_ftobInc_seg1 = np.min(seg1IncTimes[seg1IncTimes > ftobIncTimes[0]])
      next_ftobDec_seg2 = np.min(seg2DecTimes[seg2DecTimes > ftobIncTimes[0]])
      if len(next_ftobDec_seg1) > 0 and len(next_ftobInc_seg2) > 0 and next_ftobInc_seg2 < next_ftobDec_seg1:
        duration = next_ftobDec_seg1 - ftobDecTimes[0]
        overlap = next_ftobDec_seg1 - next_ftobInc_seg2
        percent_overlap[0] = overlap/duration

        ftobDecTimes[1] = next_ftobInc_seg2
        next_ftobInc_seg2 = np.min(seg2IncTimes[seg2IncTimes > ftobDecTimes[1]])
        next_ftobDec_seg3 = np.min(seg3DecTimes[seg3DecTimes > ftobDecTimes[1]])
        if len(next_ftobDec_seg2) > 0 and len(next_ftobInc_seg3) > 0 and next_ftobInc_seg3 < next_ftobDec_seg2:
          duration = next_ftobDec_seg2 - ftobDecTimes[1]
          overlap = next_ftobDec_seg2 - next_ftobInc_seg3
          percent_overlap[1] = overlap/duration

          ftobDecTimes[2] = next_ftobInc_seg3
          next_ftobDec_seg3 = np.min(seg3DecTimes[seg3DecTimes > ftobDecTimes[2]])
          next_ftobDec_seg4 = np.min(seg4DecTimes[seg4DecTimes > ftobDecTimes[2]])
          if len(next_ftobDec_seg3) > 0 and len(next_ftobInc_seg4) > 0 and next_ftobInc_seg4 < next_ftobDec_seg3:
            duration = next_ftobDec_seg3 - ftobDecTimes[2]
            overlap = next_ftobDec_seg3 - next_ftobDec_seg4
            percent_overlap[2] = overlap/duration

            ftobDecTimes[3] = next_ftobDec_seg4
            next_ftobDec_seg4 = np.min(seg4DecTimes[seg4DecTimes > ftobDecTimes[3]])
            next_ftobDec_seg5 = np.min(seg5DecTimes[seg5DecTimes > ftobDecTimes[3]])
            if len(next_ftobDec_seg4) > 0 and len(next_ftobInc_seg5) > 0 and next_ftobInc_seg5 < next_ftobDec_seg4:
              duration = next_ftobDec_seg4 - ftobDecTimes[3]
              overlap = next_ftobDec_seg4 - next_ftobDec_seg5
              percent_overlap[3] = overlap/duration

              ftobDecTimes[4] = next_ftobDec_seg5
              next_ftobDec_seg5 = np.min(seg5DecTimes[seg5DecTimes > ftobDecTimes[4]])
              next_ftobDec_seg6 = np.min(seg6DecTimes[seg6DecTimes > ftobDecTimes[4]])
              if len(next_ftobDec_seg5) > 0 and len(next_ftobDec_seg6) > 0 and next_ftobDec_seg6 < next_ftobDec_seg5:
                duration = next_ftobDec_seg5 - ftobDecTimes[4]
                overlap = next_ftobDec_seg5 - next_ftobDec_seg6
                percent_overlap[4] = overlap/duration

                ftobDecTimes[5] = next_ftobDec_seg6

      checkAltPeriod = seg1DecTimes[ix+1]
      if len(ftobDecTimes) == 6:
        ftobDecCount = ftobDecCount + 1
        ftobDecPropPeriod = ftobDecTimes[5] - ftobDecTimes[0]
        ftobDecPropPeriods = np.append(ftobDecPropPeriods, ftobDecPropPeriod)
        ftobDecCheckAltPeriods = np.append(ftobDecCheckAltPeriods, checkAltPeriod - ftobDecTimes[ix])
        ftobDecRec[:, ix] = ftobDecTimes.T
        ftobDec_percet_overlap[:, ix] = percent_overlap.T
        ftobDecTimesDiff = np.diff(ftobDecTimes)
        ftobDecPhaseDiffs[:, ix] = ftobDecTimesDiff/ftobDecPropPeriod
        allForwardPhaseDiffs = np.append(allForwardPhaseDiffs, ftobDecPhaseDiffs[:, ix])
        allForwardTimeDiffs = np.append(allForwardTimeDiffs, ftobDecTimesDiff.T)
    
    # btof inc
    for ix in np.arange(0, len(seg6IncInds)):
      percent_overlap = np.array([])
      btofIncTimes = np.array([])
      btofIncTimes[0] = seg6IncTimes(ix)
      next_btofDec_seg6 = np.min(seg6DecTimes[seg6DecTimes > btofIncTimes[0]])
      next_btofInc_seg5 = np.min(seg5IncTimes[seg5IncTimes > btofIncTimes[0]])
      if len(next_btofDec_seg6) > 0 and len(next_btofInc_seg5) > 0 and next_btofInc_seg5 < next_btofDec_seg6:
        duration = next_btofDec_seg6 - btofIncTimes[0]
        overlap = next_btofDec_seg6 - next_btofInc_seg5
        percent_overlap[0] = overlap/duration

        btofIncTimes[1] = next_btofInc_seg5
        next_btofDec_seg5 = np.min(seg5DecTimes[seg5DecTimes > btofIncTimes[1]])
        next_btofInc_seg4 = np.min(seg4IncTimes[seg4IncTimes > btofIncTimes[1]])
        if len(next_btofDec_seg5) > 0 and len(next_btofInc_seg4) > 0 and next_btofInc_seg4 < next_btofDec_seg5:
          duration = next_btofDec_seg5 - btofIncTimes[1]
          overlap = next_btofDec_seg5 - next_btofInc_seg4
          percent_overlap[1] = overlap/duration

          btofIncTimes[2] = next_btofInc_seg4
          next_btofDec_seg4 = np.min(seg4DecTimes[seg4DecTimes > btofIncTimes[2]])
          next_btofInc_seg3 = np.min(seg3IncTimes[seg3IncTimes > btofIncTimes[2]])
          if len(next_btofDec_seg4) > 0 and len(next_btofInc_seg3) > 0 and next_btofInc_seg3 < next_btofDec_seg4:
            duration = next_btofDec_seg4 - btofIncTimes[2]
            overlap = next_btofDec_seg4 - next_btofInc_seg3
            percent_overlap[2] = overlap/duration

            btofIncTimes[3] = next_btofInc_seg3
            next_btofDec_seg3 = np.min(seg3DecTimes[seg3DecTimes > btofIncTimes[3]])
            next_btofInc_seg2 = np.min(seg2IncTimes[seg2IncTimes > btofIncTimes[3]])
            if len(next_btofDec_seg3) > 0 and len(next_btofInc_seg2) > 0 and next_btofInc_seg2 < next_btofDec_seg3:
              duration = next_btofDec_seg3 - btofIncTimes[3]
              overlap = next_btofDec_seg3 - next_btofInc_seg2
              percent_overlap[3] = overlap/duration

              btofIncTimes[4] = next_btofInc_seg2
              next_btofDec_seg2 = np.min(seg2DecTimes[seg2DecTimes > btofIncTimes[4]])
              next_btofInc_seg1 = np.min(seg1IncTimes[seg1IncTimes > btofIncTimes[4]])
              if len(next_btofDec_seg2) > 0 and len(next_btofInc_seg1) > 0 and next_btofInc_seg1 < next_btofDec_seg2:
                duration = next_btofDec_seg2 - btofIncTimes[4]
                overlap = next_btofDec_seg2 - next_btofInc_seg1
                percent_overlap[4] = overlap/duration

                btofIncTimes[5] = next_btofInc_seg1

      checkAltPeriod = seg6IncTimes[ix+1]
      if len(btofIncTimes) == 6:
        btofIncCount = btofIncCount + 1
        btofIncPropPeriod = btofIncTimes[5] - btofIncTimes[0]
        btofIncPropPeriods = np.append(btofIncPropPeriods, btofIncPropPeriod)
        btofIncCheckAltPeriods = np.append(btofIncCheckAltPeriods, checkAltPeriod - btofIncTimes[ix])
        btofIncRec[:, ix] = btofIncTimes.T
        btofInc_percet_overlap[:, ix] = percent_overlap.T
        btofIncTimesDiff = np.diff(btofIncTimes)
        btofIncPhaseDiffs[:, ix] = btofIncTimesDiff/btofIncPropPeriod
        allBackwardPhaseDiffs = np.append(allBackwardPhaseDiffs, btofIncPhaseDiffs[:, ix])
        allBackwardTimeDiffs = np.append(allBackwardTimeDiffs, btofIncTimesDiff.T)
    
    # btof dec
    for ix in np.arange(0, len(seg6DecInds)):
      percent_overlap = np.array([])
      btofDecTimes = np.array([])
      btofDecTimes[0] = seg6DecTimes(ix)
      next_btofInc_seg6 = np.min(seg6IncTimes[seg6IncTimes > btofIncTimes[0]])
      next_btofDec_seg5 = np.min(seg5DecTimes[seg5DecTimes > btofIncTimes[0]])
      if len(next_btofInc_seg6) > 0 and len(next_btofDec_seg5) > 0 and next_btofDec_seg5 < next_btofInc_seg6:
        duration = next_btofInc_seg6 - btofDecTimes[0]
        overlap = next_btofInc_seg6 - next_btofDec_seg5
        percent_overlap[0] = overlap/duration

        btofDecTimes[1] = next_btofDec_seg5
        next_btofDec_seg5 = np.min(seg5DecTimes[seg5DecTimes > btofIncTimes[1]])
        next_btofInc_seg4 = np.min(seg4IncTimes[seg4IncTimes > btofIncTimes[1]])
        if len(next_btofDec_seg5) > 0 and len(next_btofInc_seg4) > 0 and next_btofInc_seg4 < next_btofDec_seg5:
          duration = next_btofDec_seg5 - btofIncTimes[1]
          overlap = next_btofDec_seg5 - next_btofInc_seg4
          percent_overlap[1] = overlap/duration

          btofDecTimes[2] = next_btofInc_seg4
          next_btofDec_seg4 = np.min(seg4DecTimes[seg4DecTimes > btofIncTimes[2]])
          next_btofInc_seg3 = np.min(seg3IncTimes[seg3IncTimes > btofIncTimes[2]])
          if len(next_btofDec_seg4) > 0 and len(next_btofInc_seg3) > 0 and next_btofInc_seg3 < next_btofDec_seg4:
            duration = next_btofDec_seg4 - btofIncTimes[2]
            overlap = next_btofDec_seg4 - next_btofInc_seg3
            percent_overlap[2] = overlap/duration

            btofDecTimes[3] = next_btofInc_seg3
            next_btofDec_seg3 = np.min(seg3DecTimes[seg3DecTimes > btofIncTimes[3]])
            next_btofInc_seg2 = np.min(seg2IncTimes[seg2IncTimes > btofIncTimes[3]])
            if len(next_btofDec_seg3) > 0 and len(next_btofInc_seg2) > 0 and next_btofInc_seg2 < next_btofDec_seg3:
              duration = next_btofDec_seg3 - btofIncTimes[3]
              overlap = next_btofDec_seg3 - next_btofInc_seg2
              percent_overlap[3] = overlap/duration

              btofDecTimes[4] = next_btofInc_seg2
              next_btofDec_seg2 = np.min(seg2DecTimes[seg2DecTimes > btofIncTimes[4]])
              next_btofInc_seg1 = np.min(seg1IncTimes[seg1IncTimes > btofIncTimes[4]])
              if len(next_btofDec_seg2) > 0 and len(next_btofInc_seg1) > 0 and next_btofInc_seg1 < next_btofDec_seg2:
                duration = next_btofDec_seg2 - btofIncTimes[4]
                overlap = next_btofDec_seg2 - next_btofInc_seg1
                percent_overlap[4] = overlap/duration

                btofDecTimes[5] = next_btofInc_seg1

      checkAltPeriod = seg6DecTimes[ix+1]
      if len(btofDecTimes) == 6:
        btofDecCount = btofDecCount + 1
        btofDecPropPeriod = btofDecTimes[5] - btofDecTimes[0]
        btofDecPropPeriods = np.append(btofDecPropPeriods, btofDecPropPeriod)
        btofDecCheckAltPeriods = np.append(btofDecCheckAltPeriods, checkAltPeriod - btofDecTimes[ix])
        btofDecRec[:, ix] = btofDecTimes.T
        btofDec_percet_overlap[:, ix] = percent_overlap.T
        btofDecTimesDiff = np.diff(btofDecTimes)
        btofDecPhaseDiffs[:, ix] = btofDecTimesDiff/btofDecPropPeriod
        allBackwardPhaseDiffs = np.append(allBackwardPhaseDiffs, btofDecPhaseDiffs[:, ix])
        allBackwardTimeDiffs = np.append(allBackwardTimeDiffs, btofDecTimesDiff.T)
    
    if len(ftobIncCheckAltPeriods) > 0:
      ftobIncPropAltRatio = ftobIncPropPeriods/ftobIncCheckAltPeriods
    else:
      ftobIncPropAltRatio = np.nan
    if len(ftobDecCheckAltPeriods) > 0:
      ftobDecPropAltRatio = ftobDecPropPeriods/ftobDecCheckAltPeriods
    else:
      ftobDecPropAltRatio = np.nan
    if len(btofIncCheckAltPeriods) > 0:
      btofIncPropAltRatio = btofIncPropPeriods/btofIncCheckAltPeriods
    else:
      btofIncPropAltRatio = np.nan
    if len(btofDecCheckAltPeriods) > 0:
      btofDecPropAltRatio = btofDecPropPeriods/btofDecCheckAltPeriods
    else:
      btofDecPropAltRatio = np.nan
    
    forwardScore1 = np.std(allForwardPhaseDiffs)
    backwardScore1 = np.std(allBackwardPhaseDiffs)
    forwardScore2 = np.std(allForwardPhaseDiffs)/np.mean(allForwardPhaseDiffs)
    backwardScore2 = np.std(allBackwardPhaseDiffs)/np.mean(allBackwardPhaseDiffs)
    forwardScore3 = np.std(allForwardTimeDiffs)/np.mean(allForwardTimeDiffs)
    backwardScore3 = np.std(allBackwardTimeDiffs)/np.mean(allBackwardTimeDiffs)

    if input == 'AVB':
      if len(allForwardTimeDiffs) > 6:
        score1 = forwardScore1
        score2 = forwardScore2
        score3 = forwardScore3
        IncCount = ftobIncCount
        DecCount = ftobDecCount
        IncPropPeriod = np.median(ftobIncPropPeriods)
        DecPropPeriod = np.median(ftobDecPropPeriods)
        IncAltPeriod = np.median(ftobIncCheckAltPeriods)
        DecAltPeriod = np.median(ftobDecCheckAltPeriods)
        IncPropAltRatio = np.median(ftobIncPropAltRatio)
        DecPropAltRatio = np.median(ftobDecPropAltRatio)
      else:
        score1 = np.nan
        score2 = np.nan
        score3 = np.nan
        IncCount = np.nan
        DecCount = np.nan
        IncPropPeriod = np.nan
        DecPropPeriod = np.nan
        IncAltPeriod = np.nan
        DecAltPeriod = np.nan
        IncPropAltRatio = np.nan
        DecPropAltRatio = np.nan
    if input == 'AVA':
      if len(allBackwardTimeDiffs) > 6:
        score1 = backwardScore1
        score2 = backwardScore2
        score3 = backwardScore3
        IncCount = btofIncCount
        DecCount = btofDecCount
        IncPropPeriod = np.median(btofIncPropPeriods)
        DecPropPeriod = np.median(btofDecPropPeriods)
        IncAltPeriod = np.median(btofIncCheckAltPeriods)
        DecAltPeriod = np.median(btofDecCheckAltPeriods)
        IncPropAltRatio = np.median(btofIncPropAltRatio)
        DecPropAltRatio = np.median(btofDecPropAltRatio)
      else:
        score1 = np.nan
        score2 = np.nan
        score3 = np.nan
        IncCount = np.nan
        DecCount = np.nan
        IncPropPeriod = np.nan
        DecPropPeriod = np.nan
        IncAltPeriod = np.nan
        DecAltPeriod = np.nan
        IncPropAltRatio = np.nan
        DecPropAltRatio = np.nan
  return np.concatenate(score1, score2, score3, IncCount, DecCount,IncPropPeriod, DecPropPeriod, IncAltPeriod, DecAltPeriod, IncPropAltRatio, DecPropAltRatio)