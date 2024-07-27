import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve
import matplotlib.pyplot as plt

def load_wav_file(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data.astype(float)

def apply_smoothing_filter(data, filter_size=256):
    kernel = np.ones(filter_size) / filter_size
    return convolve(data, kernel, mode='same')

def plot_original_and_smoothed(sample_rate, original_data, smoothed_data, duration=5):
    # Calculate the number of samples to plot
    num_samples = min(len(original_data), int(duration * sample_rate))
    
    # Create a time array for plotting
    time = np.arange(num_samples) / sample_rate
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(time, original_data[:num_samples], label='Original', alpha=0.7)
    plt.plot(time, smoothed_data[:num_samples], label='Smoothed', alpha=0.7)
    
    plt.title('Original vs Smoothed Audio Signal')
    plt.xlabel('Time ')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Specify the path to your WAV file
    file_path = 'data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav'
    
    # Load the WAV file
    sample_rate, data = load_wav_file(file_path)
    
    # If the audio is stereo, take only the first channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Apply smoothing filter
    smoothed_data = apply_smoothing_filter(data)
    
    # Plot original and smoothed data
    plot_original_and_smoothed(sample_rate, data, smoothed_data)

if __name__ == "__main__":
    main()