import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class OvarianDataset(Dataset):

  def __init__(self, csv_path, data_path, length, transforms = None):
    self.csv_p = csv_path
    self.data_p = data_path

    self.l = length
    self.transform = transforms

    self.f = open(self.csv_p)
    self.reader = csv.reader(self.f, delimiter=',')
    self.img_list = []
    self.lab_list = []

    self.num = 0
    for i in self.reader:
      self.num = self.num + 1
      if self.num == self.l:
        break
      self.img_list.append(data_path + i[0].rsplit('\\')[1])
      self.lab_list.append(np.array(i[1:]).astype(np.float32))





  def __getitem__(self, index):
    img = Image.open(self.img_list[index])



    if self.transform:
      return self.transform(img), self.lab_list[index]

    return img, self.lab_list[index]

  def __len__(self):
    return len(self.img_list)
