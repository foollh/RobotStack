import torch
import numpy as np
import json
from network import MyModel
from datasets import robotDataSets
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

def show_results(image, detected_keypoints):
    pass

def decentraliztion(keypointsNorm, imgShape):
    imgKeypoints = []
    for ikp in keypointsNorm:
        ikp[0], ikp[1] = ikp[0]*imgShape[1], ikp[1]*imgShape[0] 
        imgKeypoints.append([ikp[0], ikp[1]])

    return np.array(imgKeypoints)


def draw_annotations(raw_image, annotations):
    plt.figure("image")
    draw = ImageDraw.Draw(raw_image)
    ttf = ImageFont.load_default()
    radius = 3
    for idx, kp in enumerate(annotations["keypoints"]):
        kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
        kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
        # print("the drawn keypoints name: {}".format(keypointsNameList[idx]))
        # draw.text(kp_right_down, keypointsNameList[idx], font=ttf, fill=(255,0,0))
        draw.ellipse((kp_left_up, kp_right_down), fill=(255, 0, 0))
    
    # draw.ellipse(((293, 156), (299, 162)), fill=(0, 255, 0))

    bbox_min = annotations["bounding_box"]["min"]
    bbox_max = annotations["bounding_box"]["max"]

    draw.rectangle([bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]], outline="red", width=2)
    # draw.rectangle()
    plt.imshow(raw_image)
    plt.show()

if __name__ == "__main__":
    # function draw_annotations() test
    raw_image_path = "/home/lihua/Desktop/Datasets/DREAM/synthetic/000001.rgb.jpg"
    raw_image = Image.open(raw_image_path).convert('RGB')
    annotations = {}

    keypointsNameList = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
    
    keypointsPath = "/home/lihua/Desktop/Datasets/DREAM/synthetic/000001.json"

    keypointsNameSorted = []
    imgKeypointsList = []
    with open(keypointsPath, "r") as f:
        file = json.load(f)

    keypoints = file['objects'][0]['keypoints']
    bounding_box = file['objects'][0]['bounding_box']

    for kp in keypoints:
        if kp['name'] in keypointsNameList:
            imgKeypointsList.append((kp['projected_location']))

            keypointsNameSorted.append(kp['name'])

    annotations["bounding_box"] = bounding_box
    annotations["keypoints"] = imgKeypointsList

    draw_annotations(raw_image, annotations)

    # ################################################################## 
    # idx = 500
    # dataPath = "/home/lihua/Desktop/Datasets/DREAM/real/panda-3cam_kinect360/"

    # rds = robotDataSets(dataPath, "panda")
    # with torch.no_grad():
    #     model = MyModel("resnet50", 14, pretrained=True) 
    #     model.load_state_dict(torch.load('./VisionAlgo/trained_model/resnet50.pkl'))
    #     model.eval()

    #     # image, keypointsNorm = rds.__getitem__(idx)
    #     imgPIL = Image.open("/home/lihua/Desktop/Datasets/DREAM/real/panda-orb/000399.rgb.jpg")
    #     # image = self.transform(imgPIL)
    #     image = transforms.ToTensor()(imgPIL)[:3]

    #     detected_keypoints = model(torch.unsqueeze(image, dim=0))
    #     detected_keypoints = torch.squeeze(detected_keypoints, dim=0)
    #     # print("keypointsNorm:\n", keypointsNorm)
    #     print("detected_keypointsNorm:\n", detected_keypoints)


    #     image = transforms.ToPILImage()(image)
    #     draw = ImageDraw.Draw(image)

    #     # for idx in range(int(len(keypointsNorm)/2)):
    #     #     circle_start = (int(keypointsNorm[2*idx]*image.size[0])-3, int(keypointsNorm[2*idx+1]*image.size[1])-3)
    #     #     circle_end = (int(keypointsNorm[2*idx]*image.size[0])+3, int(keypointsNorm[2*idx+1]*image.size[1])+3)
    #     #     draw.ellipse((circle_start, circle_end), fill=(255, 0, 0))
    #         # print("keypoint{0}: {1}".format(idx, circle))

    #     for idx in range(int(len(detected_keypoints)/2)):
    #         circle_start = (int(detected_keypoints[2*idx]*image.size[0])-3, int(detected_keypoints[2*idx+1]*image.size[1])-3)
    #         circle_end = (int(detected_keypoints[2*idx]*image.size[0])+3, int(detected_keypoints[2*idx+1]*image.size[1])+3)
    #         draw.ellipse((circle_start, circle_end), fill=(0, 255, 0))

    #     plt.figure("image")
    #     plt.imshow(image)
    #     plt.show()


        # keypoints = decentraliztion(keypointsNorm.reshape((-1, 2)), (image.shape[1], image.shape[2]))
        # detected_keypoints = decentraliztion(torch.reshape(detected_keypoints, (-1, 2)), (image.shape[1], image.shape[2]))
        # # print("keypoints:\n", keypoints)
        # # print("detected_keypoints:\n", detected_keypoints)

        # img_show = (np.array(torch.permute(image, (1, 2, 0))))*255
        
        # cv2.imwrite('TestImage.png', img_show)
        # img_show = cv2.imread('TestImage.png')
        
        # for i, point in enumerate(keypoints):
        #     cv2.circle(img_show, (int(point[0]), int(point[1])), 2, (255, 0, 0), 2)
        #     cv2.circle(img_show, (int(detected_keypoints[i][0]), int(detected_keypoints[i][0])), 2, (0, 0, 255), 2)
        # cv2.imwrite('TestImage.png', img_show)
    