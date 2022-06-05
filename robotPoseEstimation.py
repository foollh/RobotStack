import time
import math
from cv2 import trace
import numpy as np
import torch
import cv2
import pybullet as pb
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from RobotStacks import Args, robotEnvironment, roboticMoving, transPixelToWorldCoordinate, transWorldToPixelCoordinate
from VisionAlgo.model_inference import resnet_inference, keypoint_rcnn_inference

# link0=link1 link4=link5 link7=link8
# keyPoints =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# ['panda_link2', 'panda_link3', 'panda_link4', 'panda_link7', 'panda_hand', 'panda_link0', 'panda_link6'] = [1, 2, 3, 5, 6, 0, 4]
keyPoints = [0, 2, 3, 4, 6, 7]
keyPointsPositions = []

# imagePoints = np.array([[119.6864, 307.9333],
#          [186.9782, 198.1121],
#          [215.2031, 214.4511],
#          [363.3090, 202.8344],
#          [378.9746, 229.2313],
#          [120.2357, 435.5816],
#          [343.7277, 216.7737]])

def rotationMatrixToEulerAngles(R) :
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([z, y, x])

def drawPoints(posIndex, debugLineLen, text, color):
    pre_pos1 = [posIndex[0]-debugLineLen, posIndex[1], posIndex[2]]
    tar_pos1 = [posIndex[0]+debugLineLen, posIndex[1], posIndex[2]]
    pre_pos2 = [posIndex[0], posIndex[1]-debugLineLen, posIndex[2]]
    tar_pos2 = [posIndex[0], posIndex[1]+debugLineLen, posIndex[2]]
    pre_pos3 = [posIndex[0], posIndex[1], posIndex[2]-debugLineLen]
    tar_pos3 = [posIndex[0], posIndex[1], posIndex[2]+debugLineLen]

    pb.addUserDebugLine(pre_pos1, tar_pos1,lineColorRGB=color, lineWidth=300)
    pb.addUserDebugLine(pre_pos2, tar_pos2,lineColorRGB=color, lineWidth=300)
    pb.addUserDebugLine(pre_pos3, tar_pos3,lineColorRGB=color, lineWidth=300)
    pb.addUserDebugText(text, pre_pos1, textColorRGB=[1, 0, 0], textSize=2.)

if __name__ == "__main__":
    args = Args()

    # camera param
    args.cameraPos = [0.4, -0.6, 0.4]
    args.cameraFocus = [0.4, 0., 0.4]
    args.cameraVector = [0., 0., 1.]
    args.cameraFov = 90
    args.cameraAspect = 640/480
    args.cameraNearVal = 0.1
    args.cameraFarVal = 20

    # args.robotInitAngle = [math.pi/9, math.pi/4.-0.3, 0.0, -math.pi/2. + 0.3, 0.0, 3*math.pi/4., -math.pi/4., -math.pi/2., -math.pi/2., 1, 1, 0]
    
    # model params set
    pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/resnet50220407_210209.pkl"
    # pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/keypoint_rcnn_04_21_15_49.pkl"
    raw_image_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/img/PandaRobotPose.jpg"
    env = robotEnvironment(args)
    # print("camera intrinsics matrix:\n", env.intrinsicsMatrix)
    env.basic_env(args)

    # cubeUid = env.regularCubes(args)
    # env.randomCubes(args)
    rm = roboticMoving(args)
    # print("base position:\n", pb.getBasePositionAndOrientation(rm.robotId))
    
    # basePos = pb.getBasePositionAndOrientation(rm.robotId)[0]
    # keyPointsPositions.append(basePos)
    # keyPointsPositions.append([0., 0., 0.])
    # start stack
    state_durations=[0.5, 0.4, 0.2, 0.5]  # the simulate time in every motion
    pb.setTimeStep=args.control_dt

    state_t=0.
    current_state=0
    debugLineLen = 0.01
    while True:
        state_t+=args.control_dt
        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                width=320, height=240,
                viewMatrix=env.viewList,
                projectionMatrix=env.projectionList,
                renderer = pb.ER_BULLET_HARDWARE_OPENGL)

        if current_state == 0:
            rm.robotPosInit(args.robotInitAngle)
            
        if current_state == 2:
            # rm.robotId = pb.loadURDF(args.robotPath, basePosition=[0, -1, 0], useFixedBase=True)
            # rm.robotPosInit(args.robotInitAngle)
            pass
            # rm.setpos(11, [0.55, -0.1, 0.072], [0, math.pi, 0.])

        if state_t>state_durations[current_state]:
            if current_state == 0:

                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=args.cameraImgWidth, height=args.cameraImgHight,
                    viewMatrix=env.viewList,
                    projectionMatrix=env.projectionList,
                    renderer = pb.ER_BULLET_HARDWARE_OPENGL)
                    # renderer = pb.ER_TINY_RENDERER)
                
                cv2.imwrite("./img/PandaRobotPose.jpg", rgbImg)

                # get image keypoints from network model
                raw_image = Image.open(raw_image_path).convert('RGB')
                ###########################################" resnet "##############################################
                _, recovered_keypoints_batch, keypointsNameList = resnet_inference(pretrained_model_path, raw_image)
                recovered_keypoints_batch = recovered_keypoints_batch.numpy()
                imagePoints = recovered_keypoints_batch[0, [5, 0, 1, 2, 6, 3, 4], :][1:, :]
                # ########################################" keypoint_rcnn "###########################################
                # detection_result, keypointsNameList = keypoint_rcnn_inference(pretrained_model_path, 'keypoint_rcnn', raw_image)
                # recovered_keypoints_batch = detection_result['keypoints'][None, :, :2]
                # imagePoints = recovered_keypoints_batch[0, [5, 0, 1, 2, 6, 3, 4], :][1:, :]
                # Draw keypoints in the image
                plt.figure("image")
                draw = ImageDraw.Draw(raw_image)
                ttf = ImageFont.load_default()
                radius = 3
                # add illustration
                draw.text((1, 1), 'Green points: ground truth', font=ttf, fill=(255,0,0))
                draw.text((1, 10), 'Blue points: inference', font=ttf, fill=(255,0,0))
                

                for keypoints in recovered_keypoints_batch:
                    for idx, kp in enumerate(keypoints):
                        kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
                        kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
                        # print("the drawn keypoints name: {}".format(keypointsNameList[idx]))
                        draw.text(kp_right_down, keypointsNameList[idx], font=ttf, fill=(255,0,0))
                        draw.ellipse((kp_left_up, kp_right_down), fill=(0, 0, 255))
                        # break
                # raw_image.save("detected_panda.jpg")
                # # plt.imshow(raw_image)

                # objectPointsInfer = transPixelToWorldCoordinate(width, height, imagePoints, depthImg, env.projectionMatrix, env.viewMatrix)
                
                # for i, point in enumerate(objectPointsInfer):
                #     drawPoints(point, debugLineLen, keypointsNameList[i], [1, 0, 0])

                for idx in range(12):
                    if idx in keyPoints:
                        keyPointsPositions.append(rm.getpos(idx)[0])
                        rm.drawDebugPoint(idx, debugLineLen, [0, 1, 0])

                keypointsImagePos = transWorldToPixelCoordinate(keyPointsPositions, env.intrinsicsMatrix, env.extrinsicMatrix)
                print("keypointImagePos:\n", np.array(keypointsImagePos))

                # Draw keypoints in the image
                for idx, kp in enumerate(keypointsImagePos):
                    kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
                    kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
                    # print("the drawn keypoints: \n {} {}".format(kp_left_up, kp_right_down))
                    # draw.text(kp_right_down, keypointsNameList[idx], font=ttf, fill=(255,0,0))
                    draw.ellipse((kp_left_up, kp_right_down), fill=(0, 255, 0))
                    # break
                raw_image.save("./img/detected_panda1.jpg")
                # plt.imshow(raw_image)

                objectPoints = np.array(keyPointsPositions)
                print("image keypoints:\n", imagePoints)
                # print("objectPointsInfer:\n", objectPointsInfer)
                print("objectPoints:\n", objectPoints)

                _, R, T = cv2.solvePnP(objectPoints=objectPoints,
                                        imagePoints=imagePoints,
                                        cameraMatrix=env.intrinsicsMatrix,
                                        distCoeffs=env.distCoeffs,
                                        flags=cv2.SOLVEPNP_EPNP)
                print('所求结果:')
                RT = np.eye(4)
                RT[:3, :3] = cv2.Rodrigues(R)[0]
                RT[:3, -1] = T.reshape(3)
                print("RT transform:\n", RT)
                # print("RT inv transform:\n", np.linalg.inv(RT))
                print("camera extrinsics matrix:\n", env.extrinsicMatrix)  
                # print("camera inv extrinsics matrix:\n", np.linalg.inv(env.extrinsicMatrix))  
                
                # calculate the error
                diff_r = RT[:3, :3] @ env.extrinsicMatrix[:3, :3]
                # np.acos((trace(diff_r)-1)/2)*180/np.pi

                # angle_error = (np.pi - np.arccos((np.sum(trace(diff_r))-1)/2))*180/np.pi
                angle_error = np.arccos((np.sum(trace(diff_r))-1)/2)
                trans_error = np.sum(np.abs(RT[:3, -1] - env.extrinsicMatrix[:3, -1]))
                print("calibration angle error: \t", angle_error)
                print("calibration translation error: \t", trans_error)

            if current_state == 1:
                pass
                # pb.removeBody(rm.robotId)
            #     cubeBasePos, cubeBaseOrn = pb.getBasePositionAndOrientation(cubeUid)
            #     print("cubeBasePos:\n", cubeBasePos)
            #     print("cubeBaseOrn:\n", cubeBaseOrn)

            current_state+=1
            if current_state>=len(state_durations):
                break
                current_state = 3
            state_t=0

        pb.stepSimulation()