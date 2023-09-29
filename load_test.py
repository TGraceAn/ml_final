from CustomAudioDataset import TrainingAudio
from torch.utils.data import DataLoader

training_data = TrainingAudio("data")
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)

# for waveform, sample_rate, label in train_dataloader:
#     print(waveform.shape)
#     print(sample_rate)
#     print(label)