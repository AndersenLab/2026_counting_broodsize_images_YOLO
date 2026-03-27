# Scoping
# Louter = []
# for i in range(5):
#     L = [i,i+1,i+2]
#     Louter.append(L)

# print(Louter)

# dict enum
# D = {"a":1,"b":2}
# for i,key in enumerate(D):
#     print(i,key)

import numpy as np
import sounddevice as sd
import time

def list_audio_devices():
    print("Available audio devices:")
    print(sd.query_devices())

list_audio_devices()

def play_sound(frequency, duration, pause, device, volume):
    """Play a sound at a given frequency and duration, then pause for a specified time, on a specific device."""
    fs = 44100  # Sampling frequency

    # Generate the sound wave
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)

    # Normalize to 16-bit range
    tone *= 32767 / np.max(np.abs(tone))
    
    # Apply volume
    tone *= volume

    # Convert to integer
    tone = tone.astype(np.int16)

    while True:
        print("Playing sound...")
        # Play the sound on the specified device
        sd.play(tone, fs, device=device)
        sd.wait()

        # Pause
        time.sleep(pause)

# Example usage
device_id = 6 # Replace with the ID of your desired output device
play_sound(250, 1, 4, device_id, volume = 1.0)

