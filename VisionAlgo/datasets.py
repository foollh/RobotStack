import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


class robotDataSets(Dataset):
    def __init__(self, dataPath, robotType) -> None:
        super(robotDataSets, self).__init__()

        self.cameraIntrinsicMatrix = np.eye(3)
        self.dataPath = dataPath

        if robotType == "panda":
            self.fileList, self.imgExtension, self.configExtension = self.pandaDatasetLoader(dataPath)

        self.len = len(self.fileList)

    def pandaDatasetLoader(self, dataPath):
        fileList = []
        for file in os.listdir(dataPath):
            path = os.path.join(dataPath, file)
            fileSplit = os.path.splitext(path)
            if fileSplit[-1] == '.jpg':
                fileList.append(file.split('.')[0])

        return fileList, '.rgb.jpg', '.json'

    def getKeypointsFromPath(self, index):
        imgKeypointsList = []
        keypointsNameList = []
        keypointsPath = self.dataPath + self.fileList[index] + self.configExtension
        with open(keypointsPath, "r") as f:
            file = json.load(f)
        keypoints = file['objects'][0]['keypoints']
        for kp in keypoints:
            imgKeypointsList.append(kp['projected_location'])
            keypointsNameList.append(kp['name'])
        
        return imgKeypointsList, keypointsNameList

    def __getitem__(self, index):
        imgKeypointsList, keypointsNameList = self.getKeypointsFromPath(index)
        img = cv2.imread(self.dataPath + self.fileList[index] + self.imgExtension)
        # for i, keypoint in enumerate(imgKeypointsList):
        #     cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), 3, (255, 0, 0), -1)
        #     cv2.putText(img, keypointsNameList[i], (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.imwrite("testImg.png", img)
        return img, np.array(imgKeypointsList)

    def __len__(self):
        return self.len





if __name__ == "__main__":
    dataPath = "/home/lihua/Desktop/Datasets/DREAM/real/panda-3cam_kinect360/"

    rds = robotDataSets(dataPath, "panda")

    temp = rds.__getitem__(0)

    # fileName = "000000.json"
    # sampleFile = os.path.join(dataPath, fileName)
    # with open(sampleFile, "r") as f:
    #     file = json.load(f)
    # basePos = file['objects'][0]['location']
    # keypoints = file['objects'][0]['keypoints']

    # print(file['objects'])