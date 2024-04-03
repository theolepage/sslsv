import numpy as np
import scipy.signal


class VAD:
    
    def __init__(self, threshold=-50, win_len=0.025, win_hop=0.025):
        self.threshold = threshold
        self.win_len = win_len
        self.win_hop = win_hop
    
    def _frame_signal(self, signal, sr):
        assert self.win_len >= self.win_hop

        frame_length = int(self.win_len * sr)
        frame_step = int(self.win_hop * sr)
        signal_length = len(signal)
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_signal = np.append(signal, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))

        nrows = ((pad_signal.size - frame_length) // frame_step) + 1
        n = pad_signal.strides[0]
        frames = np.lib.stride_tricks.as_strided(
            pad_signal,
            shape=(nrows, frame_length),
            strides=(frame_step * n, n)
        )
        return frames
    
    def apply(self, audio, sr=16000):
        frames = self._frame_signal(audio, sr)
        nb_frames, frames_len = frames.shape

        energy = np.sum(np.abs(np.fft.rfft(a=frames, n=nb_frames)) ** 2, axis=-1) / (nb_frames ** 2)
        log_energy = 10 * np.log10(energy / 1e7)

        energy = scipy.signal.medfilt(log_energy, 5)

        energy = np.repeat(energy, frames_len)

        vad = energy > self.threshold

        output = frames.flatten()[vad]
        
        self.last_vad = vad
        self.last_energy = energy
        
        return output