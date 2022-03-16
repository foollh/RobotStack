import time
import math
import numpy as np
import cv2
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving

keyPoints = [0, 2, 3, 4, 6, 8]
keyPointsPositions = []

imagePoints = np.array([[117.23585205, 443.09146729],
 [119.11187744, 309.89703369],
 [187.75427246, 199.22590942],
 [219.63129883, 213.79416504],
 [342.67485352, 214.08239136],
 [361.93344727, 199.84606934],
 [380.48967285, 226.82374878]])

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

def drawPoints(posIndex, debugLineLen, color):
    pre_pos1 = [posIndex[0]-debugLineLen, posIndex[1], posIndex[2]]
    tar_pos1 = [posIndex[0]+debugLineLen, posIndex[1], posIndex[2]]
    pre_pos2 = [posIndex[0], posIndex[1]-debugLineLen, posIndex[2]]
    tar_pos2 = [posIndex[0], posIndex[1]+debugLineLen, posIndex[2]]
    pre_pos3 = [posIndex[0], posIndex[1], posIndex[2]-debugLineLen]
    tar_pos3 = [posIndex[0], posIndex[1], posIndex[2]+debugLineLen]

    pb.addUserDebugLine(pre_pos1, tar_pos1,lineColorRGB=color, lineWidth=300)
    pb.addUserDebugLine(pre_pos2, tar_pos2,lineColorRGB=color, lineWidth=300)
    pb.addUserDebugLine(pre_pos3, tar_pos3,lineColorRGB=color, lineWidth=300)
    # pb.addUserDebugText(str(idx), pre_pos1, textColorRGB=[1, 0, 0], textSize=2.)

if __name__ == "__main__":
    args = Args()

    # camera param
    args.cameraPos = [0.65, 0., 1.0]
    args.cameraFocus = [0.65001, 0., 0.]
    args.cameraVector = [0., 0., -1.]
    args.cameraFov = 90
    args.cameraAspect = 640/480
    args.cameraNearVal = 0.1
    args.cameraFarVal = 20

    env = robotEnvironment(args)
    # print("camera intrinsics matrix:\n", env.intrinsicsMatrix)
    env.basic_env(args)

    cubeUid = env.regularCubes(args)
    # env.randomCubes(args)
    rm = roboticMoving(args)
    # print("base position:\n", pb.getBasePositionAndOrientation(rm.robotId))
    
    basePos = pb.getBasePositionAndOrientation(rm.robotId)[0]
    # keyPointsPositions.append([0., 0., 0.])
    keyPointsPositions.append(basePos)
    
    # start stack
    state_durations=[0.2, 0.5, 0.2, 0.5]  # the simulate time in every motion
    pb.setTimeStep=args.control_dt

    state_t=0.
    current_state=0
    debugLineLen = 0.01
    while True:
        state_t+=args.control_dt
        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                width=320, height=240,
                viewMatrix=env.viewMatrix,
                projectionMatrix=env.projectionMatrix)

        if current_state == 0:
            rm.robotPosInit(args.robotInitAngle)
            
        # if current_state == 1:
        #     rm.setpos(11, [0.55, -0.1, 0.072], [0, math.pi, 0.])

        if state_t>state_durations[current_state]:
            if current_state == 0:
                pb.removeBody(rm.robotId)
                # [0.55, -0.1, 0.1]

                drawPoints([0.55, -0.1, 0.025], debugLineLen, color=[1, 0, 0])
                drawPoints([0.65, -0., 0.025], debugLineLen, color=[1, 0, 0])
                time.sleep(0.1)
                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=args.cameraImgWidth, height=args.cameraImgHight,
                    viewMatrix=env.viewMatrix,
                    projectionMatrix=env.projectionMatrix,
                    renderer = pb.ER_TINY_RENDERER)
                
                # cv2.imwrite("testPandaRobot.png", rgbImg)
                # ''''''
                # for idx in range(12):
                #     if idx in keyPoints:
                #         keyPointsPositions.append(rm.getpos(idx)[0])
                #     if idx == 8:
                #         rm.drawDebugPoint(8, debugLineLen, [1, 0, 0])
                #     elif idx == 7:
                #         pass
                #     else:
                #         rm.drawDebugPoint(idx, debugLineLen, [0, 1, 0])
                
                # objectPoints = np.array(keyPointsPositions)
                # _, R, T = cv2.solvePnP(objectPoints=objectPoints, imagePoints=imagePoints, cameraMatrix=env.intrinsicsMatrix, distCoeffs=env.distCoeffs)
                # print('所求结果:')
                # print("旋转向量",R)
                # print("旋转矩阵", cv2.Rodrigues(R)[0])
                # print("欧拉角\n", rotationMatrixToEulerAngles(cv2.Rodrigues(R)[0]))
                # print("平移向量",T)
                # '''''''    

            # if current_state == 2:
            #     # pb.removeBody(rm.robotId)
            #     cubeBasePos, cubeBaseOrn = pb.getBasePositionAndOrientation(cubeUid)
            #     print("cubeBasePos:\n", cubeBasePos)
            #     print("cubeBaseOrn:\n", cubeBaseOrn)

            current_state+=1
            if current_state>=len(state_durations):
                # break
                current_state = 2
            state_t=0

        pb.stepSimulation()