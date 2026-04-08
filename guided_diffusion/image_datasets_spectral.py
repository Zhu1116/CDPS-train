import scipy.io
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_type,
    data_dir,
    batch_size,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    image_path = data_dir + '/' + data_type + '.mat'
    dataset = ImageDataset(image_path)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
        self,
        image_path,
    ):
        super().__init__()
        self.img = scipy.io.loadmat(image_path)['LRHSI'].astype(np.float32) * 2 - 1  # load data and change the data range to -1~1
        self.resolution = self.img.shape[0]

    def __len__(self):
        return self.resolution * self.resolution

    def __getitem__(self, idx):
        row = idx // self.resolution
        col = idx % self.resolution

        spectral_values = self.img[row, col, :]

        return spectral_values, {}
