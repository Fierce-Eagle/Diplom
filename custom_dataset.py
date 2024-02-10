import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_frame):
        """
        Класс обработки датасета
        :param data_frame: датасет вида pd.DataFrame({data: , label: })
        """
        self.data_frame = data_frame


    def __len__(self):
        """
        Число всех данных в датасете
        :return:
        """
        return len(self.data_frame["data"])

    def __getitem__(self, idx):
        data = self.data_frame["data"][idx]
        label = self.data_frame["label"][idx]
        return torch.Tensor([data]), torch.Tensor([label])
