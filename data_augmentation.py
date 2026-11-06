import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert
import librosa

def new_freq_lti_optimized205(y20, sr):
    # Use float32 to cut memory in half
    y20 = y20.astype(np.float32)

    # Time vector (float32)
    t = np.arange(len(y20), dtype=np.float32) / sr

    # Analytic signal (complex64)
    analytic = hilbert(y20).astype(np.complex64)

    # Envelope (complex64)
    env = analytic * np.exp(-1j * 2 * np.pi * 20000 * t).astype(np.complex64)

    # Precompute exponentials once
    exp205 = np.exp(1j * 2 * np.pi * 20500 * t).astype(np.complex64)

    # Allocate output once
    combined = np.zeros(len(y20), dtype=np.float32)

    # Accumulate in-place
    combined += np.real(env * exp205).astype(np.float32)

    # Normalize in-place
    combined /= np.max(np.abs(combined)) + 1e-12

    return combined
def new_freq_lti_optimized195(y20, sr):
    # Use float32 to cut memory in half
    y20 = y20.astype(np.float32)

    # Time vector (float32)
    t = np.arange(len(y20), dtype=np.float32) / sr

    # Analytic signal (complex64)
    analytic = hilbert(y20).astype(np.complex64)

    # Envelope (complex64)
    env = analytic * np.exp(-1j * 2 * np.pi * 20000 * t).astype(np.complex64)

    # Precompute exponentials once
    exp195 = np.exp(1j * 2 * np.pi * 19500 * t).astype(np.complex64)

    # Allocate output once
    combined = np.zeros(len(y20), dtype=np.float32)

    # Accumulate in-place
    combined += np.real(env * exp195).astype(np.float32)

    # Normalize in-place
    combined /= np.max(np.abs(combined)) + 1e-12

    return combined

def time_stretch_fixed_sr(x, sr, warp_factor):

    #librosa time_stretch function for temporal scaling
    return librosa.effects.time_stretch(x, rate=warp_factor)


def add_band_limited_noise(x, sr, snr_db,low=19300, high=20700):


    x = np.asarray(x, dtype=np.float64)

    #bandpass filter


    sos = signal.butter( 6, [low, high], btype="bandpass", fs=sr, output="sos")

  
    #Filter original signal

    x_band = signal.sosfilt(sos, x)

    sig_power = np.mean(x_band ** 2)


    #generate noise
    noise = np.random.normal(loc=0.0, scale=1.0,size=len(x))


    noise_band = signal.sosfilt(sos, noise)

    noise_power = np.mean(noise_band ** 2)


    desired_noise_power = (sig_power /(10 ** (snr_db / 10)))

    if sig_power <= 0 or noise_power <= 0:
        return x.copy()

    noise_band *= np.sqrt(desired_noise_power / noise_power)

    #add noise and return

    return x + noise_band


def pitch_shift_analytic(x, sr, shift_hz):

    #Analytic signal: removes negative frequencies
    x_a = hilbert(x)

    #exponential shift
    t = np.arange(len(x)) / sr
    osc = np.exp(1j * 2 * np.pi * shift_hz * t)
    y_a = x_a * osc

    y = np.real(y_a)
    return y.astype(np.float32)