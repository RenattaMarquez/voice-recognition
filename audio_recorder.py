import os
import sounddevice as sd
from scipy.io.wavfile import write
import datetime

class AudioRecorder:
    def __init__(self, folder="command", sample_rate=16000, channels=1, duration=3):
        self.folder = folder
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration

        os.makedirs(self.folder, exist_ok=True)

    def record(self, filename=None):
        print(f"Recording for {self.duration} seconds at {self.sample_rate}Hz, mono...")

        # Start recording
        recording = sd.rec(int(self.duration * self.sample_rate),
                           samplerate=self.sample_rate,
                           channels=self.channels,
                           dtype='int16')
        sd.wait()
        print("Recording finished.")

        # Generate filename if not provided
        if filename is None:
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"

        filepath = os.path.join(self.folder, filename)

        # Save audio
        write(filepath, self.sample_rate, recording)
        print(f"Saved to: {filepath}")
        return filepath
