from data import OvarianDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch
from AlexNet import AlexNet
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

print(torch.__version__)

file = '/media/dsigpu5/SSD/YUANHAN/CClassification/data/ovarian3000Keys.csv'

tsfm = transforms.Compose([transforms.Resize((128, 128)),
                           # transforms.RandomVerticalFlip(0.5),
                           # transforms.RandomHorizontalFlip(0.5),
                           # transforms.RandomRotation((-45,45)),
                           transforms.ToTensor(),
                           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])
odata = OvarianDataset(csv_path = file, length = 5000, data_path = '/media/dsigpu5/SSD/YUANHAN/CClassification/data/temp/',transforms=tsfm)
train_loader = DataLoader(dataset = odata, batch_size = 32, shuffle = True, num_workers = 2)



net = AlexNet().cuda(0)



print('training data length =', len(train_loader))
optimizer = optim.Adam(net.parameters(), lr = 0.1)
criterion = nn.MSELoss().cuda(0)
net.train()

for epoch in range(100):
  for i, data in enumerate(train_loader):
    optimizer.zero_grad()
    input, labels = data
    print(input)
    batch_x, batch_y = Variable(input.type('torch.FloatTensor').cuda(0)), Variable(labels.type('torch.FloatTensor').cuda(0))

    # print(batch_x.max(), batch_y.max())
    outputs = net(batch_x)
    loss = criterion(outputs, batch_y)
  print(loss)