import time
import random
import math
import numpy as np
import cv2
from cv2 import trace
import pybullet as pb
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from RobotStacks import Args, robotEnvironment, roboticMoving, transDepthBufferToRealZ, transWorldToPixelCoordinate
from VisionAlgo.model_inference import resnet_inference, keypoint_rcnn_inference

def robot_pose_estimation(env, rm, rgbimg_path):
    raw_image = Image.open(rgbimg_path).convert('RGB')

    ###########################################" resnet "##############################################
    pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/resnet50220407_210209.pkl"
    _, recovered_keypoints_batch, keypointsNameList = resnet_inference(pretrained_model_path, raw_image)
    recovered_keypoints_batch = recovered_keypoints_batch.numpy()

    # ########################################" keypoint_rcnn "###########################################
    # pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/keypoint_rcnn_04_21_15_49.pkl"
    # detection_result, keypointsNameList = keypoint_rcnn_inference(pretrained_model_path, 'keypoint_rcnn', raw_image)
    # recovered_keypoints_batch = detection_result['keypoints'][None, :, :2]

    imagePoints = recovered_keypoints_batch[0, [5, 0, 1, 2, 6, 3, 4], :][1:, :]
    # imagePoints = recovered_keypoints_batch[0, [5, 0, 1, 2, 6, 3, 4], :]


    plt.figure("image")
    draw = ImageDraw.Draw(raw_image)
    ttf = ImageFont.load_default()
    radius = 3

    # # boxes
    # bbox = detection_result['boxes']
    # draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)

    for keypoints in recovered_keypoints_batch:
        for idx, kp in enumerate(keypoints):
            kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
            kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
            # print("the drawn keypoints name: {}".format(keypointsNameList[idx]))
            draw.text(kp_right_down, keypointsNameList[idx], font=ttf, fill=(0,0,0))
            draw.ellipse((kp_left_up, kp_right_down), fill=(0, 0, 255))

    keyPoints = [0, 2, 3, 4, 6, 7]
    # keyPointsPositions = []
    keyPointsPositions = [(0., 0., 0.)]
    for idx in range(12):
        if idx in keyPoints:
            keyPointsPositions.append(rm.getpos(idx)[0])

    keypointsImagePos = transWorldToPixelCoordinate(keyPointsPositions, env.intrinsicsMatrix, env.extrinsicMatrix)
    print("keypointImagePos:\n", np.array(keypointsImagePos))

    # Draw keypoints in the image
    for idx, kp in enumerate(keypointsImagePos):
        kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
        kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
        draw.ellipse((kp_left_up, kp_right_down), fill=(0, 255, 0))

    raw_image.save("./img/detected_PandaRobotPose.jpg")
    
    objectPoints = np.array(keyPointsPositions)[1:, :]

    # solve pnp
    _, R, T = cv2.solvePnP(objectPoints=objectPoints,
                            imagePoints=imagePoints,
                            cameraMatrix=env.intrinsicsMatrix,
                            distCoeffs=env.distCoeffs,
                            flags=cv2.SOLVEPNP_EPNP)
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

    return RT


if __name__ == "__main__":
    args = Args()

    # camera param
    args.cameraPos = [0.5, -0.6, 0.5]
    args.cameraFocus = [0.5, 0., 0.5]
    args.cameraVector = [0., 0., 1.]
    args.cameraFov = 90
    args.cameraAspect = 640/480
    args.cameraNearVal = 0.1
    args.cameraFarVal = 20

    # args.robotInitAngle = [math.pi/9, math.pi/4.-0.3, 0.0, -math.pi/2. + 0.3, 0.0, 3*math.pi/4., -math.pi/4., -math.pi/2., -math.pi/2., 1, 1, 0]
    
    env = robotEnvironment(args)
    env.basic_env(args)

    # env.regularCubes(args)
    # env.randomCubes(args)
    rm = roboticMoving(args)

    RT = np.eye(4)

    state_durations=[0.5, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]  # the simulate time in every motion
    pb.setTimeStep=args.control_dt
    # while True:
    random_x = 0.5 * (random.random()-0.5)
    random_y = 0.5
    random_z = 0.5 + 0.6 * (random.random()-0.5)
    random_x, random_y, random_z = -0.2, 0.5, 0.8

    cubePosition=[random_x, random_y, -0]
    cube_point = np.array([[random_x], [random_y], [random_z], [1]])
    cubePosition = np.linalg.inv(env.extrinsicMatrix) @ cube_point

    cubeUid = env.singleCube(0.03, [cubePosition[0][0], cubePosition[1][0], cubePosition[2][0]])

    state_t=0.
    current_state=0
    while True:
        state_t+=args.control_dt
        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                width=320, height=240,
                viewMatrix=env.viewList,
                projectionMatrix=env.projectionList)

        # xpos = args.cubeBasePosition[0]
        # ypos = args.cubeBasePosition[1]
        # zpos = args.cubeBasePosition[2]
        if current_state == 0:
            rm.robotPosInit(args.robotInitAngle)
            # pass

        if current_state == 1:
            cube_pos_from_vision = np.linalg.inv(RT) @ cube_point
            rm.setpos(6, [cube_pos_from_vision[0][0], cube_pos_from_vision[1][0], 0.4], [0, math.pi, -math.pi/4])
            
        if current_state == 2:
            rm.setpos(6, [cube_pos_from_vision[0][0], cube_pos_from_vision[1][0], 0.2], [0, math.pi, -math.pi/4])

        if current_state == 3:
            rm.gripperPick()

        if current_state == 4:
            rm.setpos(6, [cube_pos_from_vision[0][0], cube_pos_from_vision[1][0], 0.4], [0, math.pi, -math.pi/4])

        if current_state == 5:
            rm.setpos(6, [0.6, 0., 0.4], [0, math.pi, -math.pi/4])
        
        if current_state == 6:
            rm.setpos(6, [0.6, 0., 0.21], [0, math.pi, -math.pi/4])

        if current_state == 7:
            rm.gripperPush()
        
        if current_state == 8:
            rm.setpos(6, [0.6, 0., 0.4], [0, math.pi, -math.pi/4])
        
        if current_state == 9:
            rm.robotPosInit(args.robotInitAngle)

        if state_t>state_durations[current_state]:
            if current_state == 0:
                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                                width=args.cameraImgWidth, height=args.cameraImgHight,
                                viewMatrix=env.viewList,
                                projectionMatrix=env.projectionList,
                                renderer = pb.ER_BULLET_HARDWARE_OPENGL)
                                # renderer = pb.ER_TINY_RENDERER)
            
                cv2.imwrite("./img/PandaRobotPose.jpg", rgbImg)

                RT = robot_pose_estimation(env, rm, "./img/PandaRobotPose.jpg")
            # if current_state == 1:
            #     width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            #                     width=args.cameraImgWidth, height=args.cameraImgHight,
            #                     viewMatrix=env.viewList,
            #                     projectionMatrix=env.projectionList,
            #                     renderer = pb.ER_BULLET_HARDWARE_OPENGL)
            #                     # renderer = pb.ER_TINY_RENDERER)
            
            #     cv2.imwrite("./img/state1.jpg", rgbImg)

            # if current_state == 2:
            #     width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            #                     width=args.cameraImgWidth, height=args.cameraImgHight,
            #                     viewMatrix=env.viewList,
            #                     projectionMatrix=env.projectionList,
            #                     renderer = pb.ER_BULLET_HARDWARE_OPENGL)
            #                     # renderer = pb.ER_TINY_RENDERER)
            
            #     cv2.imwrite("./img/state2.jpg", rgbImg)

            # if current_state == 4:
            #     width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            #                     width=args.cameraImgWidth, height=args.cameraImgHight,
            #                     viewMatrix=env.viewList,
            #                     projectionMatrix=env.projectionList,
            #                     renderer = pb.ER_BULLET_HARDWARE_OPENGL)
            #                     # renderer = pb.ER_TINY_RENDERER)
            
            #     cv2.imwrite("./img/state4.jpg", rgbImg)

            # if current_state == 6:
            #     width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            #                     width=args.cameraImgWidth, height=args.cameraImgHight,
            #                     viewMatrix=env.viewList,
            #                     projectionMatrix=env.projectionList,
            #                     renderer = pb.ER_BULLET_HARDWARE_OPENGL)
            #                     # renderer = pb.ER_TINY_RENDERER)
            
            #     cv2.imwrite("./img/state6.jpg", rgbImg)

            
            # if current_state == 8:
            #     width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            #                     width=args.cameraImgWidth, height=args.cameraImgHight,
            #                     viewMatrix=env.viewList,
            #                     projectionMatrix=env.projectionList,
            #                     renderer = pb.ER_BULLET_HARDWARE_OPENGL)
            #                     # renderer = pb.ER_TINY_RENDERER)
            
            #     cv2.imwrite("./img/state8.jpg", rgbImg)
            
            # if current_state == 9:
            #     width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            #                     width=args.cameraImgWidth, height=args.cameraImgHight,
            #                     viewMatrix=env.viewList,
            #                     projectionMatrix=env.projectionList,
            #                     renderer = pb.ER_BULLET_HARDWARE_OPENGL)
            #                     # renderer = pb.ER_TINY_RENDERER)
            
            #     cv2.imwrite("./img/state9.jpg", rgbImg)

            current_state+=1
            if current_state>=len(state_durations):
                # pb.removeBody(cubeUid)
                break
            state_t=0

        pb.stepSimulation()
