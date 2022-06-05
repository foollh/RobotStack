import os
import sys
import json
import torch
import cv2
# import albumentations as albu
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def collate_fn(batch):
        return tuple(zip(*batch))


class robotDataSets(Dataset):
    def __init__(self, dataPath, robotType, networkType):
        super(robotDataSets, self).__init__()

        self.imageTransform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])

        # self.imageAugment = albu.Compose(
        #     [albu.augmentations.geometric.resize.Resize(400, 400)],
        #     keypoint_params=albu.KeypointParams(format="xy")
        # )


        # self.cameraIntrinsicMatrix = np.eye(3)

        self.dataPath = dataPath
        self.robotType = robotType
        self.networkType = networkType

        if self.robotType == "panda":
            self.fileList, self.imgExtension, self.configExtension = self.pandaDatasetLoader(dataPath)
            self.keypointsNameList = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
            
        self.len = len(self.fileList)   
    

    def pandaDatasetLoader(self, dataPath):
        fileList = []
        for file in os.listdir(dataPath):
            path = os.path.join(dataPath, file)
            fileSplit = os.path.splitext(path)
            if fileSplit[-1] == '.jpg':
                fileList.append(file.split('.')[0])

        return fileList, '.rgb.jpg', '.json'

    def getAnnotationsFromJSON(self, index):
        imgKeypointsList = []
        keypointsNameSorted = []
        keypointsPath = os.path.join(self.dataPath, self.fileList[index]) + self.configExtension
        with open(keypointsPath, "r") as f:
            file = json.load(f)
        keypoints = file['objects'][0]['keypoints']
        bounding_box = file['objects'][0]['bounding_box']
        bounding_box = [bounding_box['min'][0], bounding_box['min'][1], bounding_box['max'][0], bounding_box['max'][1]]
        for kp in keypoints:
            if kp['name'] in self.keypointsNameList:
                imgKeypointsList.append((kp['projected_location']))

                keypointsNameSorted.append(kp['name'])
        
        return imgKeypointsList, keypointsNameSorted, bounding_box

    def imagePreprocessing(self, raw_image, keypointsList, bounding_box, net_in_resolution=(400, 400), heatmaps_resolution=(200, 200)):
        # resize image
        preprocessed_image = raw_image.resize(net_in_resolution, resample=Image.BILINEAR)
        # transformed = self.imageAugment(image=raw_image, keypoints=keypointsList)

        # shrink the keypoints and bounding_box
        preprocessed_bbox = [0, 0, 0, 0]
        preprocessed_bbox[0] = bounding_box[0] / raw_image.size[0] * net_in_resolution[0]
        preprocessed_bbox[1] = bounding_box[1] / raw_image.size[1] * net_in_resolution[1]
        preprocessed_bbox[2] = bounding_box[2] / raw_image.size[0] * net_in_resolution[0]
        preprocessed_bbox[3] = bounding_box[3] / raw_image.size[1] * net_in_resolution[1]

        preprocessed_keypoints = []
        for proj in keypointsList:
            kp_netin = [
                proj[0] / raw_image.size[0] * heatmaps_resolution[0],
                proj[1] / raw_image.size[1] * heatmaps_resolution[1],
            ]
            preprocessed_keypoints.append(kp_netin)

        return preprocessed_image, preprocessed_keypoints, preprocessed_bbox

    def createHeatmap(self, image_resolution, pointsBelief, sigma=2):
        # image size (width x height)
        # list of points to draw in a 7x2 tensor
        # the size of the point
        # returns a tensor of n_points x h x w with the belief maps
        # Input argument handling
        assert (
            len(image_resolution) == 2
        ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
            len(image_resolution)
        )
        image_width, image_height = image_resolution
        image_transpose_resolution = (image_height, image_width)
        out = np.zeros((len(pointsBelief), image_height, image_width))

        w = int(sigma * 2)

        for i_point, point in enumerate(pointsBelief):
            pixel_u = round(point[0])
            pixel_v = round(point[1])
            array = np.zeros(image_transpose_resolution)

            # TODO makes this dynamics so that 0,0 would generate a belief map.
            if (
                pixel_u - w >= 0
                and pixel_u + w + 1 < image_width
                and pixel_v - w >= 0
                and pixel_v + w + 1 < image_height
            ):
                for i in range(pixel_u - w, pixel_u + w + 1):
                    for j in range(pixel_v - w, pixel_v + w + 1):
                        array[j, i] = np.exp(
                            -(
                                ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                                / (2 * (sigma ** 2))
                            )
                        )
            out[i_point] = array

        return out
    

    def pre_inference(self, raw_image):
        preprocessed_image = raw_image.resize((400, 400), resample=Image.BILINEAR)

        preprocessed_image = self.imageTransform(preprocessed_image)
        preprocessed_image_batch = torch.unsqueeze(preprocessed_image, 0)
        
        return preprocessed_image_batch


    def __getitem__(self, index):
        annotations = {}

        imgKeypointsList, keypointsNameSorted, bounding_box = self.getAnnotationsFromJSON(index)

        raw_image = Image.open(os.path.join(self.dataPath, self.fileList[index]) + self.imgExtension).convert('RGB')
        
        if bounding_box[0] < 0.0:
                bounding_box[0] = 0.0
        if bounding_box[1] < 0.0:
            bounding_box[1] = 0.0
        if bounding_box[2] > raw_image.size[0]:
            bounding_box[2] = raw_image.size[0]
        if bounding_box[3] > raw_image.size[1]:
            bounding_box[3] = raw_image.size[1]

        # raw_image = cv2.imread(os.path.join(self.dataPath, self.fileList[index]) + self.imgExtension)
        # raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        if self.networkType == "resnet":
            net_in_resolution = (400, 400)
            heatmaps_resolution = (200, 200)
            preprocessed_image, preprocessed_keypoints, preprocessed_bbox = self.imagePreprocessing(raw_image=raw_image,
                                                                                                    keypointsList=imgKeypointsList, 
                                                                                                    bounding_box=bounding_box, 
                                                                                                    net_in_resolution=net_in_resolution,
                                                                                                    heatmaps_resolution=heatmaps_resolution)
            keypoints_heatmaps = self.createHeatmap(heatmaps_resolution, preprocessed_keypoints)

            preprocessed_image = self.imageTransform(preprocessed_image)
            keypoints_heatmaps = torch.tensor(keypoints_heatmaps).float()

            for kp in imgKeypointsList:
                if 0 < kp[0] < raw_image.size[0] and 0 < kp[1] < raw_image.size[1]:
                    kp.append(1.0)
                else:
                    kp.append(0.0)

            annotations['keypoints'] = torch.tensor([imgKeypointsList], dtype=torch.float32)
            annotations['keypoints_heatmaps'] = keypoints_heatmaps
            annotations['keypoints_name'] = keypointsNameSorted
            annotations['bounding_box'] = preprocessed_bbox
            return preprocessed_image, annotations


        elif self.networkType == "keypoint_rcnn":
            annotations['boxes'] = torch.unsqueeze(torch.as_tensor(bounding_box, dtype=torch.float), 0)
            annotations['labels'] = torch.tensor([1], dtype=torch.int64)
            # annotations['image_id'] = torch.tensor([index])
            # annotations['area'] = torch.tensor((bounding_box[3]-bounding_box[1])*(bounding_box[2]-bounding_box[0]), dtype=torch.float32)
            # annotations['iscrowd'] = torch.tensor([0], dtype=torch.int64)

            for kp in imgKeypointsList:
                if 0 < kp[0] < raw_image.size[0] and 0 < kp[1] < raw_image.size[1]:
                    kp.append(1.0)
                else:
                    kp.append(0.0)

            annotations['keypoints'] = torch.tensor([imgKeypointsList], dtype=torch.float32)

            # annotations['keypoints'] = imgKeypointsList
            # annotations['keypoints_name'] = keypointsNameSorted
            return transforms.ToTensor()(raw_image), annotations

        elif self.networkType == "keypoint_rcnn_finetune":
            annotations['boxes'] = torch.unsqueeze(torch.as_tensor(bounding_box, dtype=torch.float), 0)
            annotations['labels'] = torch.tensor([1], dtype=torch.int64)
            # annotations['image_id'] = torch.tensor([index])
            # annotations['area'] = torch.tensor((bounding_box[3]-bounding_box[1])*(bounding_box[2]-bounding_box[0]), dtype=torch.float32)
            # annotations['iscrowd'] = torch.tensor([0], dtype=torch.int64)

            for kp in imgKeypointsList:
                if 0 < kp[0] < raw_image.size[0] and 0 < kp[1] < raw_image.size[1]:
                    kp.append(1.0)
                else:
                    kp.append(0.0)

            annotations['keypoints'] = torch.tensor([imgKeypointsList], dtype=torch.float32)

            # annotations['keypoints'] = imgKeypointsList
            # annotations['keypoints_name'] = keypointsNameSorted
            return transforms.ToTensor()(raw_image), annotations

        else:
            print("the dataset network type can't accepted")
            sys.exit(1)


    def __len__(self):
        return self.len


if __name__ == "__main__":
    # dataPath = "/workspace/data/panda_synth_train_dr"
    dataPath = "/home/lihua/Desktop/Datasets/DREAM/synthetic/panda_synth_train_dr"

    index = 10

    # ########################"keypoint_rcnn"##############################
    # rds_rcnn = robotDataSets(dataPath, "panda", "keypoint_rcnn")

    # image1, annotations1 = rds_rcnn.__getitem__(index)
    # imagePIL1 = transforms.ToPILImage()(image1)

    # train_loader = DataLoader(dataset=rds_rcnn, batch_size=3, shuffle=True, num_workers=8, collate_fn=collate_fn)
    # iterator = iter(train_loader)
    # batch = next(iterator)

    # # images = list(image for image in batch[0])
    # # labels = [{k: v for k, v in t.items()} for t in batch[1]]

    # # imagePIL1 = transforms.ToPILImage()(images[0])
    # # annotations1 = labels[0]
    
    # plt.figure("keypoint_rcnn_image")
    # draw1 = ImageDraw.Draw(imagePIL1)
    # radius = 2
    # for keypoints in annotations1["keypoints"].numpy()[0]:
    #     kp_left_up = (round(keypoints[0]-radius), round(keypoints[1]-radius))
    #     kp_right_down = (round(keypoints[0]+radius), round(keypoints[1]+radius))
    #     draw1.ellipse((kp_left_up, kp_right_down), fill=(255, 0, 0))

    # draw1.rectangle([annotations1['boxes'][0, 0], annotations1['boxes'][0, 1], annotations1['boxes'][0, 2], annotations1['boxes'][0, 3]], outline="red", width=3)
    # plt.imshow(imagePIL1)


    #########################"resnet"###############################
    rds_resnet = robotDataSets(dataPath, "panda", "resnet")

    image2, annotations2 = rds_resnet.__getitem__(index)
    imagePIL2 = transforms.ToPILImage()(image2)

    plt.figure("resnet_image")
    draw2 = ImageDraw.Draw(imagePIL2)
    # radius = 3
    # for keypoints in annotations2["keypoints"]:
    #     kp_left_up = (round(keypoints[0]-radius), round(keypoints[1]-radius))
    #     kp_right_down = (round(keypoints[0]+radius), round(keypoints[1]+radius))
    #     draw2.ellipse((kp_left_up, kp_right_down), fill=(255, 0, 0))
    draw2.rectangle([annotations2['bounding_box'][0], annotations2['bounding_box'][1], annotations2['bounding_box'][2], annotations2['bounding_box'][3]], outline="red", width=3)
    plt.imshow(imagePIL2)

    plt.figure("heatmap")
    for idx in range(len(annotations2["keypoints_heatmaps"])):
        plt.subplot(2, 4, idx+1)
        tem_image = transforms.ToPILImage()(annotations2["keypoints_heatmaps"][idx])
        plt.imshow(tem_image)

    ########################"keypoint_rcnn_finetune"##############################
    # rds_rcnn_finetune = robotDataSets(dataPath, "panda", "keypoint_rcnn_finetune")

    # image1, annotations1 = rds_rcnn_finetune.__getitem__(index)
    # imagePIL1 = transforms.ToPILImage()(image1)

    # # train_loader = DataLoader(dataset=rds_rcnn_finetune, batch_size=3, shuffle=True, num_workers=8, collate_fn=collate_fn)
    # # iterator = iter(train_loader)
    # # batch = next(iterator)

    # # images = list(image for image in batch[0])
    # # labels = [{k: v for k, v in t.items()} for t in batch[1]]

    # # imagePIL1 = transforms.ToPILImage()(images[0])
    # # annotations1 = labels[0]
    
    # plt.figure("keypoint_rcnn_image")
    # draw1 = ImageDraw.Draw(imagePIL1)
    # radius = 2
    # for keypoints in annotations1["keypoints"].numpy()[0]:
    #     kp_left_up = (round(keypoints[0]-radius), round(keypoints[1]-radius))
    #     kp_right_down = (round(keypoints[0]+radius), round(keypoints[1]+radius))
    #     draw1.ellipse((kp_left_up, kp_right_down), fill=(255, 0, 0))
    # bbox = torch.round(torch.squeeze(annotations1['boxes'], 0))
    # draw1.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
    # plt.imshow(imagePIL1)


    plt.show()