from torch.utils.data import Dataset
from scipy.io import wavfile
import os


class TrainingAudio(Dataset):
    def __init__(self, dir_path: str):
        self.label_list: list[str] = []
        self.file_list: list[str] = []

        for name in os.listdir(dir_path):
            if name == "_background_noise_" or os.path.isfile(os.path.join(dir_path, name)):
                continue

            folder_path = os.path.join(dir_path, name)
            for file in os.listdir(folder_path):
                self.file_list.append(os.path.join(folder_path, file))
                self.label_list.append(name)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = wavfile.read(self.file_list[idx])
        sample_rate = data[0]
        waveform = data[1]
        return waveform, sample_rate, self.label_list[idx]
