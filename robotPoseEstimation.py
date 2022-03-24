import time
import math
import numpy as np
import cv2
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving, transPixelToWorldCoordinate

keyPoints = [0, 2, 3, 4, 6, 8]
keyPointsPositions = []

imagePoints = np.array([[98.62232056, 438.90421143],
 [ 99.30222778, 309.38789063],
 [168.97767334, 199.77555542],
 [201.5012085,  213.8038147 ],
 [328.17619629, 217.60598145],
 [347.97160645, 203.90126953],
 [362.57539063, 228.3822876]])

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
    args.cameraPos = [0.55, -0.6, 0.5]
    args.cameraFocus = [0.55, 0., 0.5]
    args.cameraVector = [0., 0., 1.]
    args.cameraFov = 90
    args.cameraAspect = 640/480
    args.cameraNearVal = 0.1
    args.cameraFarVal = 20

    args.robotInitAngle = [math.pi/9, math.pi/4.-0.3, 0.0, -math.pi/2. + 0.3, 0.0, 3*math.pi/4., -math.pi/4., -math.pi/2., -math.pi/2., 1, 1, 0]
      
    env = robotEnvironment(args)
    # print("camera intrinsics matrix:\n", env.intrinsicsMatrix)
    env.basic_env(args)

    # cubeUid = env.regularCubes(args)
    # env.randomCubes(args)
    rm = roboticMoving(args)
    # print("base position:\n", pb.getBasePositionAndOrientation(rm.robotId))
    
    basePos = pb.getBasePositionAndOrientation(rm.robotId)[0]
    # keyPointsPositions.append([0., 0., 0.])
    keyPointsPositions.append(basePos)
    
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
            
        if current_state == 1:
            pass
            # rm.setpos(11, [0.55, -0.1, 0.072], [0, math.pi, 0.])

        if state_t>state_durations[current_state]:
            if current_state == 0:
                # pb.removeBody(rm.robotId)
                # [0.55, -0.1, 0.1]

                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=args.cameraImgWidth, height=args.cameraImgHight,
                    viewMatrix=env.viewList,
                    projectionMatrix=env.projectionList,
                    renderer = pb.ER_BULLET_HARDWARE_OPENGL)
                    # renderer = pb.ER_TINY_RENDERER)
                
                cv2.imwrite("PandaRobotPose.png", rgbImg)
                objectPointsInfer = transPixelToWorldCoordinate(width, height, imagePoints, depthImg, env.projectionMatrix, env.viewMatrix)
                
                for point in objectPointsInfer:
                    drawPoints(point, debugLineLen, [1, 0, 0])

                for idx in range(12):
                    if idx in keyPoints:
                        keyPointsPositions.append(rm.getpos(idx)[0])
                    # if idx == 8:
                    #     rm.drawDebugPoint(8, debugLineLen, [1, 0, 0])
                    # elif idx == 7:
                    #     pass
                    # else:
                    rm.drawDebugPoint(idx, debugLineLen, [0, 1, 0])
                
                objectPoints = np.array(keyPointsPositions)
                print("objectPointsInfer:\n", objectPointsInfer)
                print("objectPoints:\n", objectPoints)

                # _, R, T = cv2.solvePnP(objectPoints=objectPoints, imagePoints=imagePoints, cameraMatrix=env.intrinsicsMatrix, distCoeffs=env.distCoeffs)
                # print('所求结果:')
                # RT = np.eye(4)
                # RT[:3, :3] = cv2.Rodrigues(R)[0]
                # RT[:3, -1] = T.reshape(3)
                # print("RT transform:\n", RT)
                # print("RT inv transform:\n", np.linalg.inv(RT))
                # print("camera extrinsics matrix:\n", env.extrinsicMatrix)  
                # print("camera inv extrinsics matrix:\n", np.linalg.inv(env.extrinsicMatrix))  

            if current_state == 2:
                pass
                pb.removeBody(rm.robotId)
            #     cubeBasePos, cubeBaseOrn = pb.getBasePositionAndOrientation(cubeUid)
            #     print("cubeBasePos:\n", cubeBasePos)
            #     print("cubeBaseOrn:\n", cubeBaseOrn)

            current_state+=1
            if current_state>=len(state_durations):
                # break
                current_state = 3
            state_t=0

        pb.stepSimulation()