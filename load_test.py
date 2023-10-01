import torch

from CustomAudioDataset import TrainingAudio
from models.KWT import KWT
from utils.trainer import *

# training_data = TrainingAudio("data")
# train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)

# model = KWT()
# model.load_state_dict(torch.load("runs/kwt1_baseline/best.pth"))
# model.eval()

print(torch.load("runs/kwt1_baseline/best.pth"))
