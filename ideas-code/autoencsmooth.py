import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_wav_file(file_path):
    _, data = wavfile.read(file_path)
    return data.astype(np.float32)

def apply_smoothing_filter(audio, filter_size=256):
    # Create a smoothing filter
    smooth_filter = np.ones(filter_size) / filter_size
    # Apply the filter
    smoothed_audio = convolve(audio, smooth_filter, mode='same')
    return smoothed_audio

class AudioDataset(Dataset):
    def __init__(self, audio_files, max_length=None, filter_size=256):
        self.audio_files = audio_files
        self.max_length = max_length
        self.filter_size = filter_size
        self.audios = []
        
        if max_length is None:
            self.max_length = max(len(load_wav_file(file)) for file in audio_files)
        
        for file in audio_files:
            audio = load_wav_file(file)
            # Apply smoothing filter
            audio = apply_smoothing_filter(audio, self.filter_size)
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            elif len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)), 'constant')
            self.audios.append(torch.from_numpy(audio).unsqueeze(0).float())

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audios[idx]

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1)
            # Removed Tanh activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Ensure the output size matches the input size
        if decoded.size(-1) != x.size(-1):
            decoded = F.interpolate(decoded, size=x.size(-1), mode='linear', align_corners=False)
        return decoded
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def calculate_average_baseline_mse(test_loader, device):
    total_mse = 0
    total_samples = 0
    
    # Calculate the average of all audio samples
    sum_audio = 0
    count_audio = 0
    for batch in test_loader:
        sum_audio += batch.sum().item()
        count_audio += batch.numel()
    average_audio = sum_audio / count_audio
    
    # Calculate MSE using this average
    criterion = nn.MSELoss()
    for batch in tqdm(test_loader, desc="Evaluating Baseline"):
        batch = batch.to(device)
        average_prediction = torch.full_like(batch, average_audio)
        mse = criterion(average_prediction, batch)
        total_mse += mse.item() * batch.size(0)
        total_samples += batch.size(0)
    
    return total_mse / total_samples

def evaluate_autoencoder(folder_path, batch_size=8, epochs=10, filter_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
    train_files, test_files = train_test_split(audio_files, test_size=0.2, random_state=42)

    # Find the maximum length among all audio files
    max_length = max(len(load_wav_file(file)) for file in audio_files)
    
    # Ensure max_length is divisible by 8 (due to 3 layers of stride 2)
    max_length = ((max_length - 1) // 8 + 1) * 8

    train_dataset = AudioDataset(train_files, max_length, filter_size)
    test_dataset = AudioDataset(test_files, max_length, filter_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Final evaluation
    autoencoder_mse = evaluate_model(model, test_loader, criterion, device)
    
    # Calculate baseline MSE
    baseline_mse = calculate_average_baseline_mse(test_loader, device)

    print(f"Convolutional Autoencoder MSE: {autoencoder_mse}")
    print(f"Average Baseline MSE: {baseline_mse}")
    print(f"Improvement over baseline: {baseline_mse - autoencoder_mse}")

    return autoencoder_mse, baseline_mse

# Usage
folder_path = 'data'
autoencoder_mse, baseline_mse = evaluate_autoencoder(folder_path, filter_size=256)
print(f"The autoencoder outperforms the average baseline by {baseline_mse - autoencoder_mse} MSE")