import parser
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import robotDataSets



def train(model, epochs, criterion, optimizer, train_loader):
    for epoch in range(epochs):
        loss_temp = []
        for i, (imgs, labels) in enumerate(train_loader):
            # forward
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss_temp.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:{0}\tloss:{1}\t".format(epoch+1, sum(loss_temp)/len(loss_temp)))

if __name__ == "__main__":
    dataPath = "/home/lihua/Desktop/Datasets/DREAM/real/panda-3cam_kinect360/"

    rds = robotDataSets(dataPath, "panda")

    tran_loader = DataLoader(dataset=rds, batch_size=32, shuffle=True, num_workers=4)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam()

    train()
