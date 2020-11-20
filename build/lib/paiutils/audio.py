"""
Author: Travis Hammond
Version: 11_20_2019
"""


import os
import subprocess
import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct

try:
    import webrtcvad
except ModuleNotFoundError:
    print('ModuleError: webrtcvad could not be found. '
          'Therefore, vad_trim_all, vad_trim_sides, '
          'and vad_split cannot be used.')

use_pyaudio = False
try:
    import pyaudio
    use_pyaudio = True
except ModuleNotFoundError:
    print('ModuleError: pyaudio could not be found. '
          'Therefore, sox will be used for recording and playing audio.')

util_dir = os.path.dirname(__file__)

if os.name == 'nt':
    sox_path = os.path.join(util_dir, 'sox', 'sox.exe')
    if not os.path.exists(sox_path):
        print(f'SoX does not exist or is not in the '
              f'location: {sox_path}\nDownload SoX: '
              f'https://sourceforge.net/projects/sox/\n'
              f'Some functionally will be disabled until resolved.')
else:
    sox_path = None
    if sox_path is None:
        print('SoX is only configured to work with Windows, '
              'so some functionally will be disabled.')


CHUNK = 1000


def convert_width_to_atype(width, signed=True):
    """Converts a number of bytes to an audio type.
    params:
        width: An integer within 1-4 (inclusive)
        signed: A boolean, which determines if the type is signed
    return: A string, which is the audio type
    """
    if width == 1:
        atype = 'int8'
    elif width == 2:
        atype = 'int16'
    elif width == 3:
        atype = 'int24'
    elif width == 4:
        atype = 'float32'
    else:
        raise ValueError('width can only 1, 2, 3, or 4')
    if signed:
        return atype
    return 'u' + atype


def convert_atype_to_width(atype):
    """Converts an audio type to the number of bytes each value takes.
    params:
        atype: A string, which is an audio type
    return: An integer within 1-4 (inclusive)
    """
    if atype == 'int8':
        return 1
    if atype == 'uint8':
        return 1
    if atype == 'int16':
        return 2
    if atype == 'int24':
        return 3
    if atype == 'float32':
        return 4


def change_rate(audio, rate, new_rate, atype=None):
    """Changes the audio's sample rate.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        new_rate: An integer, which is the rate to change the audio to
    return: A tuple of the new audio and rate
    """
    if rate == new_rate:
        return audio, rate
    temp_filename = os.path.join(
            util_dir, str(np.random.randint(10000, 100000))+'.wav'
    )
    save(temp_filename, audio, rate, atype=atype)
    try:
        cmd = [sox_path, temp_filename, '-r ' + str(new_rate), temp_filename]
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        with wave.open(temp_filename, 'r') as file:
            atype = convert_width_to_atype(file.getsampwidth())
            rate = file.getframerate()
            audio = file.readframes(file.getnframes())
            audio = np.frombuffer(audio, dtype=atype) / np.iinfo(atype).max
    finally:
        os.remove(temp_filename)
    return audio, rate


def load(filename, rate=None, assert_mono=True):
    """Changes the audio's sample rate.
    params:
        filename: A string, which is the directory or filename of the
                  file to load
        rate: An integer, which is the rate at which samples are taken
        assert_mono: A boolean, which determines if an assertion error
                     should be raise if there are more than one channel
                     in the audio or if it should be converted to one
                     channel
    return: A tuple of the loaded audio, rate, and atype
    """
    if filename.split('.')[-1] == 'wav':
        file = wave.open(filename, 'r')
        assert file.getnchannels() == 1 or not assert_mono, (
            'Can only load mono-channel files'
        )
    if (filename.split('.')[-1] == 'wav'
            and file.getnchannels() == 1
            and (rate is None or file.getframerate() == rate)):
        atype = convert_width_to_atype(file.getsampwidth())
        rate = file.getframerate()
        audio = file.readframes(file.getnframes())
        audio = np.frombuffer(audio, dtype=atype) / np.iinfo(atype).max
        file.close()
    else:
        if filename.split('.')[-1] == 'wav':
            file.close()
        temp_filename = os.path.join(
            util_dir, str(np.random.randint(10000, 100000))+'.wav'
        )
        if rate is None:
            cmd = [sox_path, filename, '-c 1', temp_filename]
        else:
            cmd = [sox_path, filename, '-r ' + str(rate), '-c 1',
                   temp_filename]
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        try:
            with wave.open(temp_filename, 'r') as file:
                atype = convert_width_to_atype(file.getsampwidth())
                rate = file.getframerate()
                audio = file.readframes(file.getnframes())
                audio = np.frombuffer(audio, dtype=atype) / np.iinfo(atype).max
        finally:
            os.remove(temp_filename)
    return audio, rate, atype


def save(filename, audio, rate, atype=None):
    """Saves the audio to a file.
    params:
        filename: A string, which is the directory or filename of the
                  file to load
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        atype: A string, which is the audio type (default: int16)
    """
    if atype is None:
        atype = 'int16'
    with wave.open(filename, 'wb') as file:
        file.setframerate(rate)
        file.setnchannels(1)
        file.setsampwidth(convert_atype_to_width(atype))
        audio = (audio * np.iinfo(atype).max).astype(atype)
        file.writeframes(audio.tobytes())


def file_record(filename, seconds, rate, atype=None,
                recording_device_name='Microphone'):
    """Records audio from the recording device to a file.
    params:
        filename: A string, which is the directory or filename of the
                  file to load
        seconds: A float, which is the length of the recording
        rate: An integer, which is the rate at which samples are taken
        atype: A string, which is the audio type (default: int16)
        recording_device_name: A string, which is the name of the
                               recording device
    """
    if atype is None:
        atype = 'int16'
    cmd = [sox_path, f'-b {convert_atype_to_width(atype) * 8}',
           '-c 1', f'-r {rate}', f'-t waveaudio {recording_device_name}']
    if atype[0] == 'u':
        cmd += ['-e unsigned-integer']
    elif atype[0] == 'i':
        cmd += ['-e signed-integer']
    elif atype[0] == 'f':
        cmd += ['-e floating-point']
    cmd += [f'"{filename}"']
    cmd += [f'trim 0 {seconds}']
    subprocess.run(' '.join(cmd), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def record(seconds, rate, atype=None, recording_device_name='Microphone'):
    """Records audio from the recording device.
    params:
        seconds: A float, which is the length of the recording
        rate: An integer, which is the rate at which samples are taken
        atype: A string, which is the audio type (default: int16)
        recording_device_name: A string, which is the name of the
                               recording device
    return: A tuple of the loaded audio, rate, and atype
    """
    global CHUNK, use_pyaudio
    if use_pyaudio:
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate, 
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        for i in range(0, int(rate / CHUNK * seconds)):
            frames.append(stream.read(CHUNK))

        stream.stop_stream()
        stream.close()
        p.terminate()
        audio = np.frombuffer(b''.join(frames), dtype=np.int16) / np.iinfo(np.int16).max
        atype = 'int16'
    else:
        temp_filename = os.path.join(
            util_dir, str(np.random.randint(10000, 100000))+'.wav'
        )
        file_record(temp_filename, seconds, rate, atype=atype,
                    recording_device_name=recording_device_name)
        try:
            with wave.open(temp_filename, 'r') as file:
                atype = convert_width_to_atype(file.getsampwidth())
                rate = file.getframerate()
                audio = file.readframes(file.getnframes())
                audio = np.frombuffer(audio, dtype=atype) / np.iinfo(atype).max
        finally:
            os.remove(temp_filename)
    return audio, rate, atype


def file_play(filename):
    """Plays the audio file.
    params:
        filename: A string, which is the directory or filename of the
                  file to load
    """
    cmd = [sox_path, f'"{filename}"', '-t waveaudio']
    subprocess.run(' '.join(cmd), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def play(audio, rate, atype=None):
    """Plays the audio.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        atype: A string, which is the audio type (default: int16)
    """
    global CHUNK, use_pyaudio
    if use_pyaudio:
        p = pyaudio.PyAudio()
        
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate, 
                        output=True)
        
        data = np.array_split((audio * np.iinfo(np.int16).max).astype(np.int16), CHUNK)
        for frame in data:
            stream.write(frame.tobytes())
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    else:
        temp_filename = os.path.join(
            util_dir, str(np.random.randint(10000, 100000))+'.wav'
        )
        audio = np.pad(audio, (0, rate), 'constant')
        save(temp_filename, audio, rate, atype=atype)
        try:
            file_play(temp_filename)
        finally:
            os.remove(temp_filename)


def calc_duration(audio, rate):
    """Calculates the length of the audio in seconds.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
    return: A float
    """
    return audio.size / rate


def set_length(audio, length, mode='R', pad_value=0):
    """Sets the length of audio.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        length: An integer, which is the length to set the audio to
        mode: A string ('L','R','B'), which determines where to pad or remove
        pad_values: A float within -1.0 to 1.0 (inclusive), which will be
                    the if the audio is padded
    return: A numpy ndarray, which has 1 dimension and values within
            -1.0 to 1.0 (inclusive)
    """
    mode = mode.lower()
    assert mode in 'lbr', 'mode must be L(Left), R(Right), or B(Both)'
    size = audio.size
    if size > length:
        if mode == 'l':
            return audio[size-length:]
        elif mode == 'r':
            return audio[:length-size]
        else:
            return audio[(size-length)//2:-(size-length)//2]
    else:
        if mode == 'l':
            return np.pad(audio, (length-size, 0),
                          'constant', constant_values=pad_value)
        elif mode == 'r':
            return np.pad(audio, (0, length-size),
                          'constant', constant_values=pad_value)
        else:
            return np.pad(audio, ((length-size)//2, (length-size+1)//2),
                          'constant', constant_values=pad_value)


def set_duration(audio, rate, seconds, mode='R', pad_value=0):
    """Sets the duration of audio in seconds.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        seconds: A float, which is the duration to set the audio to
        mode: A string ('L','R','B'), which determines where to pad or remove
        pad_values: A float within -1.0 to 1.0 (inclusive), which will be
                    the value if the audio is padded
    return: A numpy ndarray, which has 1 dimension and values within
            -1.0 to 1.0 (inclusive)
    """
    return set_length(audio, round(rate * seconds), mode, pad_value)


def for_each_frame(audio, rate, frame_duration, func):
    """Calls a function on each frame.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
        func: A function, which takes a frame and returns a value
    return: A tuple of a numpy ndarray of results from func and integer
            (new rate)
    """
    frames = np.array_split(
        audio, int(audio.size / (rate * frame_duration))
    )
    audio = np.array([func(frame) for frame in frames])
    return audio, round(1 / frame_duration)


def compute_spectrogram(audio, rate, frame_duration, real=True):
    """Computes a nonoverlapping spectrogram.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
        real: A boolean, which determines if one side hermitian ffts
              should be used or real ffts
    return: A tuple of a numpy ndarray, which has 2 dimensions
            (frame, frequency powers), and an integer (new rate)
    """
    if real:
        def ft(frame):
            x = np.fft.hfft(frame)
            return x[:len(x)//2]
    else:
        def ft(frame):
            return np.fft.rfft(frame)
    return for_each_frame(audio, rate, frame_duration, ft)


def convert_spectrogram_to_audio(spectrogram, rate, real=True):
    """Converts a nonoverlapping spectrogram back to audio.
    params:
        spectrogram: A numpy ndarray, which has 2 dimensions
        rate: An integer, which is the rate at which each frame is taken
        real: A boolean, which determines if one side hermitian ffts
              should be used or real ffts
    return: A tuple of a numpy ndarray, which has 1 dimension,
            and an integer (new rate)
    """
    if real:
        def ft(frame):
            frame2 = np.hstack([frame, np.flip(frame[1:-1])])
            return np.real(np.fft.ihfft(frame2))
    else:
        def ft(frame):
            return np.fft.irfft(frame)
    frames = []
    for frame in spectrogram:
        frames.append(ft(frame))
    return np.hstack(frames), len(frames[0]) * rate


def compute_fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                  nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                  winfunc=lambda x: np.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.
    Code adapted from python_speech_features, written orginally by James Lyons.

    params:
        signal: the audio signal from which to compute features.
                Should be an N*1 array
        samplerate: the sample rate of the signal we are working with, in Hz.
        winlen: the length of the analysis window in seconds. Default is
                0.025s (25 milliseconds)
        winstep: the step between successive windows in seconds. Default
                 is 0.01s (10 milliseconds)
        nfilt: the number of filters in the filterbank, default 26.
        nfft: the FFT size. Default is None, which uses the calculate_nfft
              function to choose the smallest size that does not drop sample data.
        lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter.
                 Default is 0.97.
        winfunc: the analysis window to apply to each frame. By default no window is applied.
                 You can use numpy window functions here e.g. winfunc=numpy.hamming
    return: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt)
            containing features. Each row holds 1 feature vector. The second return
            value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, 'highfreq is greater than samplerate / 2'
    signal = np.append(signal[0], signal[1:] - preemph * signal[:-1])

    slen = len(signal)
    frame_len = int(winlen * samplerate + .5)
    frame_step = int(winstep * samplerate + .5)
    num_frames = 1
    if slen > frame_len:
        num_frames += int(np.ceil((slen - frame_len) / frame_step))
    pad_len = int((num_frames - 1) * frame_step + frame_len)
    padded_signal = np.concatenate([signal, np.zeros(pad_len - slen)])
    frames = np.lib.stride_tricks.as_strided(
        padded_signal,
        shape=padded_signal.shape[:-1] +
              (padded_signal.shape[-1] - frame_len + 1, frame_len),
        strides=padded_signal.strides + (padded_signal.strides[-1],)
    )[::frame_step] * winfunc(frame_len)
    pspec = 1.0 / nfft * np.square(np.abs(np.fft.rfft(frames, nfft)))
    energy = np.sum(pspec, 1)
    energy = np.where(energy == 0, np.finfo(float).eps,energy)

    lowmel = 2595 * np.log10(1 + lowfreq / 700)
    highmel = 2595 * np.log10(1 + highfreq / 700)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    fft_bins = np.floor(
        (nfft + 1) * (700 * (10**(melpoints / 2595) - 1)) / samplerate
    )
    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(fft_bins[j]), int(fft_bins[j+1])):
            fbank[j,i] = (i - fft_bins[j]) / (fft_bins[j+1]-fft_bins[j])
        for i in range(int(fft_bins[j+1]), int(fft_bins[j+2])):
            fbank[j,i] = (fft_bins[j+2]-i) / (fft_bins[j+2]-fft_bins[j+1])
    feat = np.dot(pspec, fbank.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    return feat, energy


def compute_mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                 nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=0.97,
                 ceplifter=22,append_energy=True, winfunc=lambda x: np.ones((x,))):
    """Computes MFCC features from an audio signal.
    Code adapted from python_speech_features, written orginally by James Lyons.

    params:
        signal: the audio signal from which to compute features.
                Should be an N*1 array
        samplerate: the sample rate of the signal we are working with, in Hz.
        winlen: the length of the analysis window in seconds. Default is
                0.025s (25 milliseconds)
        winstep: the step between successive windows in seconds. Default
                 is 0.01s (10 milliseconds)
        numcep: the number of cepstrum to return, default 13
        nfilt: the number of filters in the filterbank, default 26.
        nfft: the FFT size. Default is None, which uses the calculate_nfft
              function to choose the smallest size that does not drop sample data.
        lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter.
                 Default is 0.97.
        ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter.
                   Default is 22.
        append_energy: if this is true, the zeroth cepstral coefficient is replaced with
                       the log of the total frame energy.
        winfunc: the analysis window to apply to each frame. By default no window is applied.
                 You can use numpy window functions here e.g. winfunc=numpy.hamming
    return: A numpy array of size (NUMFRAMES by numcep) containing features.
            Each row holds 1 feature vector.
    """
    if nfft is None:
        winlen_samples = winlen * samplerate
        nfft = 1
        while nfft < winlen_samples:
            nfft *= 2
    feat, energy = compute_fbank(signal, samplerate, winlen, winstep, nfilt,
                                 nfft, lowfreq, highfreq, preemph, winfunc)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    if ceplifter > 0:
        feat = feat * (1 + (ceplifter / 2) *
                       np.sin((np.pi / ceplifter) * np.arange(feat.shape[1])))
    if append_energy:
        feat[:,0] = np.log(energy)
    return feat


def calc_rms(audio):
    """Calculates the Root Mean Square of the audio.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
    return: A float, which is the rms of the audio
    """
    return np.sqrt(np.sum(np.square(audio)) / audio.size)


def shift_pitch(audio, rate, steps):
    """Shifts the pitch of the audio.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
    return: A numpy ndarray, which has 1 dimension
    """
    y = np.fft.rfft(audio)
    y = np.roll(y, steps)
    if steps > 0:
        y[:steps] = 0
    else:
        y[steps:] = 0
    return np.fft.irfft(y, audio.size)


def set_power(audio, power):
    """Sets the power of the audio.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        power: A float, which is the Root Mean Square to set the audio to
    return: A numpy ndarray, which has 1 dimension
    """
    return np.clip(power / calc_rms(audio) * audio, -1, 1)


def adjust_speed(audio, rate, multiplier=1):
    """Adjusts the speed of the audio and keeps the RMS power the same.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        multiplier: A float, which is the amount to adjust the relative speed
    return: A numpy ndarray, which has 1 dimension
    """
    power = calc_rms(audio)
    y = np.fft.rfft(audio)
    ns = round(audio.size / multiplier)
    return set_power(np.fft.irfft(y, ns), power)


def set_speed(audio, rate, seconds):
    """Sets the speed of the audio and keeps the RMS power the same.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        seconds: A float, which is the number of seconds the audio
                 should be set to
    return: A numpy ndarray, which has 1 dimension
    """
    power = calc_rms(audio)
    y = np.fft.rfft(audio)
    ns = round(seconds * rate)
    return set_power(np.fft.irfft(y, ns), power)


def adjust_volume(audio, multiplier=1):
    """Adjusts the volume of the audio.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        multiplier: A float, which is the amount to adjust the relative volume
    return: A numpy ndarray, which has 1 dimension
    """
    return np.clip(audio * multiplier, -1, 1)


def blend(audio1, audio2, audio1_weight=.5, audio2_weight=None):
    """Blends two audios together.
    params:
        audio1: A numpy ndarray, which has 1 dimension and values within
                -1.0 to 1.0 (inclusive)
        audio2: A numpy ndarray, which has 1 dimension and values within
                -1.0 to 1.0 (inclusive)
        audio1_weight: A float, which is the weight of audio 1
                       and should be within 0.0 and 1.0 (exclusive)
        audio2_weight: A float, which is the weight of audio 2
                       and should be within 0.0 and 1.0 (exclusive)
    return: A numpy ndarray, which has 1 dimension
    """
    if audio2_weight is None:
        audio2_weight = 1 - audio1_weight
    if audio1.size == audio2.size:
        return audio1 * audio1_weight + audio2 * audio2_weight
    elif audio1.size > audio2.size:
        audio1 = audio1 * audio1_weight
        return np.hstack((audio1[:audio2.size] + audio2_weight * audio2,
                          audio1[audio2.size:]))
    else:
        audio2 = audio2 * audio2_weight
        return np.hstack((audio1 * audio1_weight + audio2[audio1.size:],
                          audio2[audio1.size:]))


def plot(audio, seconds=0):
    """Plots the audio on a graph.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        seconds: A float, which is the number of seconds to show the plot
    """
    plt.plot(audio)
    if seconds == 0:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(seconds)
        plt.close()


def convert_audio_to_db(audio, rate, frame_duration, ref_func=lambda x: 1,
                        min_threshold=1e-10, db_threshold=80.0):
    """Converts the audio to decibels.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
        ref_func: A function, which takes a magnitude and returns a value
        min_threshold: A float, which is the minimum magnitude
        db_threshold: A float, which is the threshold for the audio
                      in decibels
    return: A tuple of a numpy ndarray, which has 1 dimension,
            and an integer (new rate)
    """
    mag, new_rate = for_each_frame(audio, rate, frame_duration,
                                   lambda sample: calc_rms(sample))
    y = 10.0 * np.log10(np.maximum(min_threshold, mag))
    y -= 10.0 * np.log10(np.maximum(min_threshold, ref_func(mag)))
    y = y if db_threshold is None else np.maximum(y, y.max() - db_threshold)
    return y, new_rate


def convert_power_to_db(power, ref_func=lambda x: 1,
                        min_threshold=1e-10, db_threshold=80.0):
    """Converts power to decibels.
    params:
        power: A numpy ndarray, which has 1 or 2 dimensions
        min_threshold: A float, which is the minimum magnitude
        db_threshold: A float, which is the threshold for the audio
                in decibels
    return: A numpy ndarray, which has 1 or 2 dimensions
    """
    mag = np.abs(power)
    y = 10.0 * np.log10(np.maximum(min_threshold, mag))
    y -= 10.0 * np.log10(np.maximum(min_threshold, ref_func(mag)))
    y = y if db_threshold is None else np.maximum(y, y.max() - db_threshold)
    return y


def convert_amplitude_to_db(amplitude, ref_func=lambda x: 1,
                            min_threshold=1e-10, db_threshold=80.0):
    """Converts amplitude to decibels.
    params:
        amplitude: A numpy ndarray, which has 1 or 2 dimensions
        min_threshold: A float, which is the minimum magnitude
        db_threshold: A float, which is the threshold for the audio
                in decibels
    return: A numpy ndarray, which has 1 or 2 dimensions
    """
    mag = np.abs(amplitude)
    y = 20.0 * np.log10(np.maximum(min_threshold, mag))
    y -= 20.0 * np.log10(np.maximum(min_threshold, ref_func(mag)))
    y = y if db_threshold is None else np.maximum(y, y.max() - db_threshold)
    return y


def trim_all(audio, rate, frame_duration, ambient_power=1e-4):
    """Trims ambient silence in the audio anywhere.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
                        to check
        ambient_power: A float, which is the Root Mean Square of ambient noise
    return: A numpy ndarray, which has 1 dimension and values within
            -1.0 to 1.0 (inclusive)
    """
    new_audio = []
    powers, fr = for_each_frame(audio, rate, frame_duration, calc_rms)
    frame_length = round(rate / fr)
    for ndx, power in enumerate(powers):
        if power > ambient_power:
            new_audio += audio[ndx*frame_length:(ndx+1)*frame_length].tolist()
    return np.array(new_audio)


def trim_sides(audio, rate, frame_duration, ambient_power=1e-4):
    """Trims ambient silence in the audio only on the sides.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
                        to check
        ambient_power: A float, which is the Root Mean Square of ambient noise
    return: A numpy ndarray, which has 1 dimension and values within
            -1.0 to 1.0 (inclusive)
    """
    powers, fr = for_each_frame(audio, rate, frame_duration, calc_rms)
    frame_length = round(rate / fr)
    start_ndx = None
    end_ndx = None
    for ndx, power in enumerate(powers):
        if power > ambient_power and start_ndx is None:
            start_ndx = ndx * frame_length
            break
    else:
        start_ndx = 0
    for ndx, power in enumerate(reversed(powers)):
        if power > ambient_power and end_ndx is None:
            end_ndx = (len(powers) - ndx) * frame_length
            break
    else:
        end_ndx = len(audio)
    if start_ndx < end_ndx:
        return audio[start_ndx:end_ndx]
    return audio


def split(audio, rate, frame_duration, ambient_power=1e-4, min_gap=None):
    """Splits the audio into audio segments on ambient frames.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
                        to check
        ambient_power: A float, which is the Root Mean Square of ambient noise
        min_gap: An integer, which is the number of frames to consider until
                 ambient frames are removed
    return: A list of numpy ndarray, which are 1 dimension each and have
            values within -1.0 to 1.0 (inclusive)
    """
    if min_gap is None:
        min_gap = frame_duration
    powers, fr = for_each_frame(audio, rate, frame_duration, calc_rms)
    frame_length = round(rate / fr)
    audios = []
    last_split = 0
    gap = 0
    on = False
    for ndx, power in enumerate(powers):
        if power > ambient_power:
            gap = 0
            on = True
        else:
            gap += frame_duration
            if gap >= min_gap and on:
                gap = 0
                on = False
                next_split = (ndx + 1) * frame_length
                audios.append(audio[last_split:next_split])
                last_split = next_split
    return audios


def find_gaps(audio, rate, frame_duration, ambient_power=1e-4):
    """Finds the length of gaps in the audio.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
                        to check
        ambient_power: A float, which is the Root Mean Square of ambient noise
    return: A list of tuples with the first value in the tuple being the start
            of a gap and the second value the end
    """
    powers, fr = for_each_frame(audio, rate, frame_duration, calc_rms)
    frame_length = round(rate / fr)
    gaps = []
    start = None
    for ndx, power in enumerate(powers):
        if power > ambient_power:
            if start is not None:
                end = (ndx + 1) * frame_length
                gaps.append((start, end))
            start = None
        elif start is None:
            start = ndx * frame_length
    if start is not None:
        gaps.append((start, len(audio)))
    return gaps


def vad_trim_all(audio, rate, frame_duration, aggressiveness=1):
    """Trims anywhere in the audio that does not contain speech.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer (8000, 16000, 32000, 48000), which is the rate
              at which samples are taken
        frame_duration: A float (.01, .02, .03), which is the duration
                        of each frame to check
        aggressiveness: A integer (0, 1, 2, 3), which is the level of
                        aggressiveness to trim non-speech
    return: A numpy ndarray, which has 1 dimension and values within
            -1.0 to 1.0 (inclusive)
    """
    assert rate in (8000, 16000, 32000, 48000), (
        'Invalid Rate, use 8000, 16000, 32000, or 48000'
    )
    assert frame_duration in (.01, .02, .03), (
        'Invalid frame_dur, use .01, .02, .03'
    )
    assert 0 <= aggressiveness <= 3, (
        'Invalid aggressiveness, must be between 0 and 3'
    )
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(rate * frame_duration)
    offset = 0
    voiced_frames = []
    while offset + frame_size < len(audio):
        frame = audio[offset:offset + frame_size]
        frame_bytes = np.int16(frame * np.iinfo('int16').max).tobytes()
        if vad.is_speech(frame_bytes, rate):
            voiced_frames.append(frame)
        offset += frame_size
    if len(voiced_frames) == 0:
        return audio
    return np.hstack(voiced_frames)


def vad_trim_sides(audio, rate, frame_duration, aggressiveness=1):
    """Trims the sides in the audio that do not contain speech.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer (8000, 16000, 32000, 48000), which is the rate
              at which samples are taken
        frame_duration: A float (.01, .02, .03), which is the duration
                        of each frame to check
        aggressiveness: A integer (0, 1, 2, 3), which is the level of
                        aggressiveness to trim non-speech
    return: A numpy ndarray, which has 1 dimension and values within
            -1.0 to 1.0 (inclusive)
    """
    assert rate in (8000, 16000, 32000, 48000), (
        'Invalid Rate, use 8000, 16000, 32000, or 48000'
    )
    assert frame_duration in (.01, .02, .03), (
        'Invalid frame_dur, use .01, .02, .03'
    )
    assert 0 <= aggressiveness <= 3, (
        'Invalid aggressiveness, must be between 0 and 3'
    )
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(rate * frame_duration)
    offset = 0
    start_ndx = 0
    while offset + frame_size < len(audio):
        frame = audio[offset:offset + frame_size]
        frame_bytes = np.int16(frame * np.iinfo('int16').max).tobytes()
        if vad.is_speech(frame_bytes, rate):
            start_ndx = offset
            break
        offset += frame_size
    else:
        return audio
    offset = len(audio)
    end_ndx = len(audio)
    while offset - frame_size > start_ndx:
        frame = audio[offset - frame_size:offset]
        frame_bytes = np.int16(frame * np.iinfo('int16').max).tobytes()
        if vad.is_speech(frame_bytes, rate):
            end_ndx = offset
            break
        offset -= frame_size
    return audio[start_ndx:end_ndx]


def vad_split(audio, rate, frame_duration, aggressiveness=1):
    """Splits the audio into audio segments on non-speech frames.
    params:
        audio: A numpy ndarray, which has 1 dimension and values within
               -1.0 to 1.0 (inclusive)
        rate: An integer, which is the rate at which samples are taken
        frame_duration: A float, which is the duration of each frame
                        to check
    return: A list of numpy ndarray, which are 1 dimension each and
            have values within -1.0 to 1.0 (inclusive)
    """
    assert rate in (8000, 16000, 32000, 48000), (
        'Invalid Rate, use 8000, 16000, 32000, or 48000'
    )
    assert frame_duration in (.01, .02, .03), (
        'Invalid frame_dur, use .01, .02, .03'
    )
    assert 0 <= aggressiveness <= 3, (
        'Invalid aggressiveness, must be between 0 and 3'
    )
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(rate * frame_duration)
    offset = 0
    off = True
    voiced_frames = []
    while offset + frame_size < len(audio):
        frame = audio[offset:offset + frame_size]
        frame_bytes = np.int16(frame * np.iinfo('int16').max).tobytes()
        if vad.is_speech(frame_bytes, rate):
            if off is True:
                off = False
                voiced_frames.append([frame])
            else:
                voiced_frames[-1].append(frame)
        else:
            off = True
        offset += frame_size
    if len(voiced_frames) == 0:
        return np.array([audio])
    for ndx in range(len(voiced_frames)):
        voiced_frames[ndx] = np.hstack(voiced_frames[ndx])
    return voiced_frames


if __name__ == '__main__':
    print('Record')
    audio, rate, atype = record(5, 32000)
    print('Play')
    play(audio, rate, atype=atype)

    print('save')
    save('saved.wav', audio, rate, atype=atype)

    print('load')
    audio2, rate2, atype2 = load('saved.wav')
    assert all(audio == audio2) and rate == rate2 and atype == atype2

    print('No overlap spectrogram')
    sg, sg_rate = compute_spectrogram(audio, rate, .1)
    plot(sg)
    print('No overlap spectrogram converted to db')
    sgdb = convert_amplitude_to_db(sg)
    plot(sgdb)

    print('convert no overlap spectrogram to audio')
    sg_audio, sg_rate2 = convert_spectrogram_to_audio(sg, sg_rate)
    play(sg_audio, sg_rate2, atype)

    print('set length left', set_length(audio, 20, mode='L').size)
    print('set length right', set_length(audio, 20, mode='R').size)
    print('set length both', set_length(audio, 20, mode='B').size)

    print('set duration left', set_duration(audio, rate, 1, mode='L').size)
    print('set duration right', set_duration(audio, rate, 1, mode='R').size)
    print('set duration both', set_duration(audio, rate, 1, mode='B').size)

    print('duration 1s',
          calc_duration(set_duration(audio, rate, 1, mode='B'), rate))

    print('for each frame avg')
    play(*for_each_frame(audio, rate, .0001, lambda x: x.mean()), atype=atype)

    print('rms', calc_rms(audio))

    print('set power to .1', calc_rms(set_power(audio, .1)))

    print('shift pitch')
    play(shift_pitch(audio, rate, 5), rate, atype=atype)

    print('adjust speed (2x)')
    play(adjust_speed(audio, rate, 2), rate, atype=atype)

    print('set speed 2s')
    play(set_speed(audio, rate, 2), rate, atype=atype)

    print('adjust volume (3x)')
    play(adjust_volume(audio, 3), rate, atype=atype)

    print('blend (echo)')
    play(blend(audio, set_speed(audio, rate, 4.5)), rate, atype=atype)

    print('audio to db')
    audio_db = convert_audio_to_db(audio, rate, .001)[0]
    plot(audio_db)

    print('trim all')
    ta = trim_all(audio, rate, .03, calc_rms(audio)/5)
    print(ta.shape, audio.shape)

    print('trim sides')
    ta = trim_sides(audio, rate, .03, calc_rms(audio)/5)
    print(ta.shape, audio.shape)

    print('split')
    ta = split(audio, rate, .03, calc_rms(audio)/8, .1)
    print(len(ta))

    print('find gaps')
    print(find_gaps(audio, rate, .03, calc_rms(audio)/5))

    power = calc_rms(audio)
    x = 1
    while True:
        gaps = find_gaps(audio, rate, .03, power/x)
        if len(gaps) == 4:  # three sounds
            ta = split(audio, rate, .03, power/x, .03)
            print(len(ta), power/x)
            break
        x += .1
        if x > 100:
            print('fail')
            break

    print('vad trim all')
    vad = vad_trim_all(audio, rate, .03, aggressiveness=3)
    print(vad.shape, audio.shape)

    print('vad trim sides')
    vad = vad_trim_sides(audio, rate, .02, aggressiveness=3)
    print(vad.shape, audio.shape)

    print('vad split')
    vad = vad_split(audio, rate, .01, aggressiveness=3)
    print(len(vad))
    if len(vad) >= 3:
        play(vad[0], rate, atype=atype)
        play(vad[1], rate, atype=atype)
        play(vad[2], rate, atype=atype)
        play(vad[2], rate, atype=atype)
        play(vad[1], rate, atype=atype)
        play(vad[0], rate, atype=atype)
    elif len(vad) == 2:
        play(vad[0], rate, atype=atype)
        play(vad[1], rate, atype=atype)
        play(vad[1], rate, atype=atype)
        play(vad[0], rate, atype=atype)
    else:
        play(vad[0], rate, atype=atype)
    print(vad[0].shape, audio.shape)
