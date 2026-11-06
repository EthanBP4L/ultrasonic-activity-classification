import gc
import ctypes
import re

import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf
import resampy
import matplotlib.pyplot as plt
import random

from data_augmentation import (
    new_freq_lti_optimized205,
    new_freq_lti_optimized195,
    time_stretch_fixed_sr,
    add_band_limited_noise,
    pitch_shift_analytic,
)


def augment_activity(segmented_carrier_shift, segment, sr=48000, methods=None):
    """
    Apply selected augmentations to one activity segment.

    Parameters
    ----------
    segment : np.ndarray
        Audio samples for one detected activity.
    sr : int
        Sampling rate.
    methods : list[str] or None
        Augmentations to apply.

    Returns
    -------
    np.ndarray
        Augmented audio segment.
    """
    if methods is None:
        methods = []
    all_augmented_segments = []
    carrier_list = []
    augmented = segment.copy()

    for method in methods:
        if method == "time_stretch":
            stretch_factor = round(
                random.uniform(0.9, 1.1), 2
            )
            all_augmented_segments.append(time_stretch_fixed_sr(
                augmented, sr, stretch_factor
            ))
            carrier_list.append(0)
           

        elif method == "noise":
            snr_db = round(random.uniform(40, 50), 2)
            all_augmented_segments.append(add_band_limited_noise(
                augmented, sr, snr_db
            ))
            carrier_list.append(0)

        elif method == "pitch_shift":
            shift_hz = round(random.uniform(-24, 24), 2)
            all_augmented_segments.append(pitch_shift_analytic(
                augmented, sr, shift_hz
            ))
            carrier_list.append(0)

        elif method == "carrier_205":
            augmented_carrier_shift = new_freq_lti_optimized205(segmented_carrier_shift, sr)

            lowcut205 = 19800.0
            highcut205 = 21200.0

            
            augmented_carrier_shift = butter_bandpass_filter(augmented_carrier_shift, lowcut205, highcut205, sr)

            lowcut205 = 20485.0
            highcut205 = 20515.0
            augmented_carrier_shift = butter_bandstop_filter(augmented_carrier_shift, lowcut205, highcut205, sr)

            all_augmented_segments.append(augmented_carrier_shift)
            carrier_list.append(1)

        elif method == "carrier_195":
            augmented_carrier_shift = new_freq_lti_optimized195(segmented_carrier_shift, sr)

            lowcut195 = 18800.0
            highcut195 = 20200.0

            
            augmented_carrier_shift = butter_bandpass_filter(augmented_carrier_shift, lowcut195, highcut195, sr)

            lowcut195 = 19485.0
            highcut195 = 19515.0
            augmented_carrier_shift = butter_bandstop_filter(augmented_carrier_shift, lowcut195, highcut195, sr)

            all_augmented_segments.append(augmented_carrier_shift)
            carrier_list.append(2)

        else:
            raise ValueError(
                f"Unknown augmentation method: {method}"
            )

    return all_augmented_segments, carrier_list


def butter_bandpass(lowcut, highcut, sr, order = 5):
  #allow only frequency between lowcut and highcut
  nyq = 0.5 * sr
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, sr):
  b, a = butter_bandpass(lowcut, highcut, sr)
  y = lfilter(b, a, data)
  return y

def butter_bandstop(lowcut, highcut, sr, order = 4):
  #get rid of frequencies between lowcut and highcut
  nyq = 0.5 * sr
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='bandstop')
  return b, a

def butter_bandstop_filter(data, lowcut, highcut, sr):
  b, a = butter_bandstop(lowcut, highcut, sr)
  y = lfilter(b, a, data)
  return y

#help with memory
def free():
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

def get_magnitudes(activity):
    # Check if activity is a file path or audio data
    # Load the audio file with sample rate 44.1kHz
    data, sr = sf.read(activity)
    #print(data.dtype)
    # Resample the audio to 48kHz
    z = resampy.resample(data, sr, 48000)
    y = z

    #y = new_freq_lti_optimized(z, sr)
    #free()
    lowcut = 19300.0
    highcut = 20700.0

    sr = 48000

    #filter frequencies outside range
    z = butter_bandpass_filter(z, lowcut, highcut, sr)

    lowcut = 19985.0
    highcut = 20015.0

    #filter frequencies inside range
    z = butter_bandstop_filter(z, lowcut, highcut, sr)

    # Number of FFT points
    #with sample rate 48khz, bin every 12 hz (rounded)
    n_fft = 4096  

    magnitude = []
    time = []
    low_freq = 19300
    high_freq = 20700

    #filter out start and end when no activity occurs 
    z = z[96000:len(z)-192000]
    #get fft and sum magnitudes to get STE for increments of 0.1s
    for i in range (0,len(z),4800):
      if (i / 48000) > -1:   # 2.2
        segment = z[i:4800+i]
        # Perform FFT
        fft_result = np.abs(np.fft.fft(segment, 4096))
        frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
        low_bin = np.where(frequencies >= low_freq)[0][0]
        high_bin = np.where(frequencies <= high_freq)[0][-1]
        # Extract the relevant frequency bins
        relevant_fft = fft_result[low_bin:high_bin]

        #STE - Short Term Energy
        magnitude.append(np.sum(relevant_fft))
        time.append((i)/48000)
    return time, magnitude, y

def split_reps(magnitude, time, activate_start=0.5):

  start_times = []
  end_times = []
  count = 0
  for i in range (len(magnitude)):

    #if significant increase in STE: start an activity
    if magnitude[i] > activate_start and len(start_times) == len(end_times):
      if i >1:
        start_times.append(time[i-1])
    if magnitude[i] > activate_start and len(start_times) > len(end_times):
      count = 0
    
    #if significant decrease for longer than 0.5s: end activity
    if magnitude[i] < activate_start and len(start_times) > len(end_times) and count > 5:
      end_times.append(time[i])
      count = 0
    if magnitude[i] < activate_start and len(start_times) > len(end_times):
      count += 1

  if len(start_times) > len(end_times):
    start_times.pop()
  return end_times, start_times

def split_reps_walk(magnitude, time):
  start_times = []
  end_times = []
  count = 0
  #walks seperated into 4 second incirimants to account for variations in walking patterns
  for i in range (len(magnitude)):
    if count == 0:
      start_times.append(time[i])
    if count < 40:
      count +=1
    if count == 40:
      end_times.append(time[i])
      count = 0

  if len(start_times) > len(end_times):
    start_times.pop()
  return end_times, start_times


def extract_features(y, start_times, end_times, label, augment = False, augmentation_methods = None):
  all_bins20 = []
  labels = []


  #these will be used for bandpass filter for LTI augmentation
  low_freq20 = 19300
  high_freq20 = 20700
  low_freq205 = 19800
  high_freq205 = 21200
  low_freq195 = 18800
  high_freq195 = 20200
  sr = 48000
  for i in range(len(start_times)):
    all_bins_temp20 = []
    #original audio
    segment20 = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]

    free()
    lowcut20 = 19300.0
    highcut20 = 20700.0

    sr = 48000
    segment20 = butter_bandpass_filter(segment20, lowcut20, highcut20, sr)

    lowcut20 = 19985.0
    highcut20 = 20015.0
    segment20 = butter_bandstop_filter(segment20, lowcut20, highcut20, sr)

    #if an activity was properly identified
    if end_times [i] - start_times[i] > 1.5 and end_times [i] - start_times[i] < 5:


      #Segment audio into 20 chunks and perform fft
      for j in range(20):

        new_segment20 = segment20[j*2400:(j+1)*2400]
        fft_result = np.abs(np.fft.fft(new_segment20, 4096))
        frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
        low_bin = np.where(frequencies >= low_freq20)[0][0]
        high_bin = np.where(frequencies <= high_freq20)[0][-1]
        relevant_fft = fft_result[low_bin:high_bin]
        all_bins_temp20.append(relevant_fft)

      all_bins_temp20 = np.array(all_bins_temp20)
      all_bins_temp20 = all_bins_temp20[:16, :118]
      all_bins20.append(all_bins_temp20)

      labels.append(label)
    
      if augment:
        segment_carrier_shift = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]
        augmented_segments, carrier_list = augment_activity(
        segment_carrier_shift,
        segment20,
        sr=48000,
        methods=augmentation_methods
        )

        for i in range (len(augmented_segments)):
          augmented_bins = []

          if carrier_list[i] == 0:
            for j in range(20):

              new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
              fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
              frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
              low_bin = np.where(frequencies >= low_freq20)[0][0]
              high_bin = np.where(frequencies <= high_freq20)[0][-1]
              relevant_fft = fft_result[low_bin:high_bin]


              augmented_bins.append(relevant_fft)

          if carrier_list[i] == 1:
            for j in range(20):

              new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
              fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
              frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
              low_bin = np.where(frequencies >= low_freq205)[0][0]
              high_bin = np.where(frequencies <= high_freq205)[0][-1]
              relevant_fft = fft_result[low_bin:high_bin]


              augmented_bins.append(relevant_fft)

          if carrier_list[i] == 2:
            for j in range(20):

              new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
              fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
              frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
              low_bin = np.where(frequencies >= low_freq195)[0][0]
              high_bin = np.where(frequencies <= high_freq195)[0][-1]
              relevant_fft = fft_result[low_bin:high_bin]


              augmented_bins.append(relevant_fft)
          augmented_bins = np.array(augmented_bins)
          augmented_bins = augmented_bins[:16, :118]
          all_bins20.append(augmented_bins)
          labels.append(label)


  return all_bins20,labels

#Same as previouse exctraction

def extract_features_walk(y, start_times, end_times, label, augment = False, augmentation_methods = None):
  all_bins20 = []

  labels = []
  low_freq20 = 19300
  high_freq20 = 20700
  low_freq205 = 19800
  high_freq205 = 21200
  low_freq195 = 18800
  high_freq195 = 20200

  sr = 48000
  for i in range(len(start_times)):

    all_bins_temp20 = []

    segment20 = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]
    free()

    lowcut20 = 19300.0
    highcut20 = 20700.0

    sr = 48000
    segment20 = butter_bandpass_filter(segment20, lowcut20, highcut20, sr)

    lowcut20 = 19985.0
    highcut20 = 20015.0
    segment20 = butter_bandstop_filter(segment20, lowcut20, highcut20, sr)
    if end_times [i] - start_times[i] > 1:
      

      for j in range(20):

        new_segment20 = segment20[j*2400:(j+1)*2400]
        fft_result = np.abs(np.fft.fft(new_segment20, 4096))
        frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
        low_bin = np.where(frequencies >= low_freq20)[0][0]
        high_bin = np.where(frequencies <= high_freq20)[0][-1]
        # Extract the relevant frequency bins
        relevant_fft = fft_result[low_bin:high_bin]
        all_bins_temp20.append(relevant_fft)

      all_bins_temp20 = np.array(all_bins_temp20)
      all_bins_temp20 = all_bins_temp20[:16, :118]
      all_bins20.append(all_bins_temp20)
      labels.append(label)

      if augment:
        segment_carrier_shift = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]
        augmented_segments, carrier_list = augment_activity(
        segment_carrier_shift,
        segment20,
        sr=48000,
        methods=augmentation_methods
        )

        for i in range (len(augmented_segments)):
          augmented_bins = []

          if carrier_list[i] == 0:
            for j in range(20):

              new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
              fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
              frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
              low_bin = np.where(frequencies >= low_freq20)[0][0]
              high_bin = np.where(frequencies <= high_freq20)[0][-1]
              relevant_fft = fft_result[low_bin:high_bin]


              augmented_bins.append(relevant_fft)

          if carrier_list[i] == 1:
            for j in range(20):

              new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
              fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
              frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
              low_bin = np.where(frequencies >= low_freq205)[0][0]
              high_bin = np.where(frequencies <= high_freq205)[0][-1]
              relevant_fft = fft_result[low_bin:high_bin]


              augmented_bins.append(relevant_fft)

          if carrier_list[i] == 2:
            for j in range(20):

              new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
              fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
              frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
              low_bin = np.where(frequencies >= low_freq195)[0][0]
              high_bin = np.where(frequencies <= high_freq195)[0][-1]
              relevant_fft = fft_result[low_bin:high_bin]


              augmented_bins.append(relevant_fft)
          augmented_bins = np.array(augmented_bins)
          augmented_bins = augmented_bins[:16, :118]
          all_bins20.append(augmented_bins)
          labels.append(label)


  return all_bins20, labels

#sit/stand is different because activities performed on the same file with pattern
#sit - stand - sit - stand, so every other activity will be a sit/stand repectively
def extract_features_sit(y, start_times, end_times, label1, label2, augment = False, augmentation_methods = None):
  all_bins20 = []

  labels = []
  low_freq20 = 19300
  high_freq20 = 20700
  low_freq205 = 19800
  high_freq205 = 21200
  low_freq195 = 18800
  high_freq195 = 20200

  sr = 48000
  for i in range(len(start_times)):

    all_bins_temp20 = []

    if end_times [i] - start_times[i] > 1.5 and end_times [i] - start_times[i] < 5:
      #if first detected activity (sit), the only difference will be the label appended
      if i%2 == 1:

        segment20 = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]
        #segment195 = new_freq_lti_optimized195(y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)], 48000)
        free()
        lowcut20 = 19300.0
        highcut20 = 20700.0

        sr = 48000
        segment20 = butter_bandpass_filter(segment20, lowcut20, highcut20, sr)

        lowcut20 = 19985.0
        highcut20 = 20015.0
        segment20 = butter_bandstop_filter(segment20, lowcut20, highcut20, sr)


        for j in range(20):

          new_segment20 = segment20[j*2400:(j+1)*2400]
          fft_result = np.abs(np.fft.fft(new_segment20, 4096))
          frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
          low_bin = np.where(frequencies >= low_freq20)[0][0]
          high_bin = np.where(frequencies <= high_freq20)[0][-1]
          # Extract the relevant frequency bins
          relevant_fft = fft_result[low_bin:high_bin]
          all_bins_temp20.append(relevant_fft)

        all_bins_temp20 = np.array(all_bins_temp20)
        all_bins_temp20 = all_bins_temp20[:16, :118]
        all_bins20.append(all_bins_temp20)

        labels.append(label1)

        if augment:
          segment_carrier_shift = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]
          augmented_segments, carrier_list = augment_activity(
          segment_carrier_shift,
          segment20,
          sr=48000,
          methods=augmentation_methods
          )

          for i in range (len(augmented_segments)):
            augmented_bins = []

            if carrier_list[i] == 0:
              for j in range(20):

                new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
                fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
                frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
                low_bin = np.where(frequencies >= low_freq20)[0][0]
                high_bin = np.where(frequencies <= high_freq20)[0][-1]
                relevant_fft = fft_result[low_bin:high_bin]


                augmented_bins.append(relevant_fft)

            if carrier_list[i] == 1:
              for j in range(20):

                new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
                fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
                frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
                low_bin = np.where(frequencies >= low_freq205)[0][0]
                high_bin = np.where(frequencies <= high_freq205)[0][-1]
                relevant_fft = fft_result[low_bin:high_bin]


                augmented_bins.append(relevant_fft)

            if carrier_list[i] == 2:
              for j in range(20):

                new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
                fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
                frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
                low_bin = np.where(frequencies >= low_freq195)[0][0]
                high_bin = np.where(frequencies <= high_freq195)[0][-1]
                relevant_fft = fft_result[low_bin:high_bin]


                augmented_bins.append(relevant_fft)
            augmented_bins = np.array(augmented_bins)
            augmented_bins = augmented_bins[:16, :118]
            all_bins20.append(augmented_bins)
            labels.append(label1)


      #this would be a stand
      elif i%2 == 0:

        segment20 = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]
        #segment195 = new_freq_lti_optimized195(y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)], 48000)
        free()
        lowcut20 = 19300.0
        highcut20 = 20700.0

        sr = 48000
        segment20 = butter_bandpass_filter(segment20, lowcut20, highcut20, sr)

        lowcut20 = 19985.0
        highcut20 = 20015.0
        segment20 = butter_bandstop_filter(segment20, lowcut20, highcut20, sr)

        for j in range(20):
          new_segment20 = segment20[j*2400:(j+1)*2400]
          fft_result = np.abs(np.fft.fft(new_segment20, 4096))
          frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
          low_bin = np.where(frequencies >= low_freq20)[0][0]
          high_bin = np.where(frequencies <= high_freq20)[0][-1]
          # Extract the relevant frequency bins
          relevant_fft = fft_result[low_bin:high_bin]
          all_bins_temp20.append(relevant_fft)

        all_bins_temp20 = np.array(all_bins_temp20)
        all_bins_temp20 = all_bins_temp20[:16, :118]
        all_bins20.append(all_bins_temp20)

        labels.append(label2)

        if augment:
          segment_carrier_shift = y[int(96000+start_times[i]*48000):int(96000+end_times[i]*48000)]
          augmented_segments, carrier_list = augment_activity(
          segment_carrier_shift,
          segment20,
          sr=48000,
          methods=augmentation_methods
          )

          for i in range (len(augmented_segments)):
            augmented_bins = []

            if carrier_list[i] == 0:
              for j in range(20):

                new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
                fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
                frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
                low_bin = np.where(frequencies >= low_freq20)[0][0]
                high_bin = np.where(frequencies <= high_freq20)[0][-1]
                relevant_fft = fft_result[low_bin:high_bin]


                augmented_bins.append(relevant_fft)

            if carrier_list[i] == 1:
              for j in range(20):

                new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
                fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
                frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
                low_bin = np.where(frequencies >= low_freq205)[0][0]
                high_bin = np.where(frequencies <= high_freq205)[0][-1]
                relevant_fft = fft_result[low_bin:high_bin]


                augmented_bins.append(relevant_fft)

            if carrier_list[i] == 2:
              for j in range(20):

                new_segment_augment = augmented_segments[i][j*2400:(j+1)*2400]
                fft_result = np.abs(np.fft.fft(new_segment_augment, 4096))
                frequencies = np.fft.fftfreq(4096, 1/sr)[:4096//2]
                low_bin = np.where(frequencies >= low_freq195)[0][0]
                high_bin = np.where(frequencies <= high_freq195)[0][-1]
                relevant_fft = fft_result[low_bin:high_bin]


                augmented_bins.append(relevant_fft)

            augmented_bins = np.array(augmented_bins)
            augmented_bins = augmented_bins[:16, :118]
            all_bins20.append(augmented_bins)
            labels.append(label2)


  return all_bins20, labels

#purely to make sure activity file is properly segmented based on activities
def plot_start_end(start_times, end_times, magnitude, time):

  plt.figure(figsize=(10, 4))

  # Base signal
  plt.plot(time, magnitude, linewidth=1.0, label="Magnitude")

  # Start markers (e.g., green)
  plt.scatter(start_times,
            [magnitude[time.index(t)] for t in start_times],
            color='green', s=20, zorder=5, label='Start Times')

  # End markers (e.g., red, larger)
  plt.scatter(end_times,
            [magnitude[time.index(t)] for t in end_times],
            color='red', s=20, zorder=6, label='End Times')

  plt.xlabel("Time")
  plt.ylabel("Magnitude")
  plt.title("Magnitude vs Time with Start/End Markers")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()

#reads audio file names and assigns class
def process_activity_file(filename, augment = False, augmentation_methods = None):

  individual_labels = []
  labels = []
  all_bins20 = []

  #the array individual_labels is used for leave one out experiments
  if re.search(r'pushup', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time, 0.5)
    #plot_start_end(start_times, end_times, magnitude, time)

    all_bins20, labels = extract_features(y, start_times, end_times, 0, augment, augmentation_methods)
    for i in range (len(labels)):
      
      individual_labels.append(filename)

  elif re.search(r'sitstand', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time, 0.5)
    #plot_start_end(start_times, end_times, magnitude, time)

    all_bins20, labels = extract_features_sit(y, start_times, end_times, 7, 8, augment, augmentation_methods)
    for i in range (len(labels)):
      
      individual_labels.append(filename)


  elif re.search(r'walkt', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps_walk(magnitude, time)
    #plot_start_end(start_times, end_times, magnitude, time)

    all_bins20, labels = extract_features_walk(y, start_times, end_times, 9, augment, augmentation_methods)
    for i in range (len(labels)):

      individual_labels.append(filename)

  elif re.search(r'walk', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps_walk(magnitude, time)
    #plot_start_end(start_times, end_times, magnitude, time)

    all_bins20, labels = extract_features_walk(y, start_times, end_times, 10, augment, augmentation_methods)
    for i in range (len(labels)):

      individual_labels.append(filename)


  elif re.search(r'latraise', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time)
    #plot_start_end(start_times, end_times, magnitude, time)
    all_bins20, labels = extract_features(y, start_times, end_times, 1, augment, augmentation_methods)
    for i in range (len(labels)):

      individual_labels.append(filename)


  elif re.search(r'lunge', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time, 0.5)
    #plot_start_end(start_times, end_times, magnitude, time)
    all_bins20, labels = extract_features(y, start_times, end_times, 2, augment, augmentation_methods)
    for i in range (len(labels)):

      individual_labels.append(filename)

  elif re.search(r'frankenstein', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time)
    #plot_start_end(start_times, end_times, magnitude, time)
    all_bins20, labels = extract_features(y, start_times, end_times, 3, augment, augmentation_methods)
    for i in range (len(labels)):

      individual_labels.append(filename)


  elif re.search(r'squat', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time)
    #plot_start_end(start_times, end_times, magnitude, time)
    all_bins20, labels = extract_features(y, start_times, end_times, 4, augment, augmentation_methods)
    for i in range (len(labels)):

      individual_labels.append(filename)


  elif re.search(r'jump', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time)
    #plot_start_end(start_times, end_times, magnitude, time)
    all_bins20, labels = extract_features(y, start_times, end_times, 5, augment, augmentation_methods)
    for i in range (len(labels)):

      individual_labels.append(filename)


  elif re.search(r'jj', filename, re.IGNORECASE):

    time, magnitude, y = get_magnitudes(filename)
    end_times, start_times = split_reps(magnitude, time)
    #plot_start_end(start_times, end_times, magnitude, time)
    all_bins20, labels = extract_features(y, start_times, end_times, 6, augment, augmentation_methods)
    for i in range (len(labels)):


      individual_labels.append(filename)


  else:
    print("No match found.")

  #verify array lengths match accordingly with labels
  print(len(labels))
  print(len(all_bins20))
  print(len(individual_labels))
  free()
  return individual_labels, labels, all_bins20

