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
        self.hr_msi = scipy.io.loadmat(image_path)['HRMSI'] * 2 - 1  # load data and change the data range to -1~1
        self.resolution = self.hr_msi.shape[0]
        self.hr_msi = np.transpose(self.hr_msi, axes=(2, 0, 1)).astype(np.float32)
        self.channel_num = self.hr_msi.shape[0]
        self.imgs = []
        # crop img for training
        step = 64
        crop_size = 128
        k = self.resolution / crop_size
        k = int(k * 2 - 1)
        self.k = k
        for ch in range(self.channel_num):
            img = self.hr_msi[ch:ch+1, :, :]
            for i in range(k):
                for j in range(k):
                    start_x = j * step
                    start_y = i * step
                    crop_img = img[:, start_y:start_y+crop_size, start_x:start_x+crop_size]
                    self.imgs.append(crop_img)

    def __len__(self):
        return self.k * self.k * self.channel_num

    def __getitem__(self, idx):
        return self.imgs[idx], {}
