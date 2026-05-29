from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CarColorDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame[["image_path", "label"]].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        item = self.frame.iloc[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(item["label"])
