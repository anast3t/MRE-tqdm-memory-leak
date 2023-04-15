import torch


class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(RandomImageDataset, self).__init__()
        self.values = torch.randn((1000, 3, 64, 64))
        self.labels = torch.randn((1000, 1))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]
