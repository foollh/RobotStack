import time
import math
import numpy as np
import cv2
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving, transDepthBufferToRealZ


if __name__ == "__main__":
    args = Args()

    # camera param
    args.cameraPos = [0.6, 0., 0.6]
    args.cameraFocus = [0.6, 0., 0.]
    args.cameraVector = [1., 0., 0.]
    args.cameraFov = 90
    args.cameraAspect = 640/480
    args.cameraNearVal = 0.1
    args.cameraFarVal = 20

    args.robotInitAngle = [math.pi/3, math.pi/4.-0.3, 0.0, -math.pi/2. + 0.3, 0.0, 3*math.pi/4., -math.pi/4., -math.pi/2., -math.pi/2., 1, 1, 0]
        

    env = robotEnvironment(args)
    env.basic_env(args)

    # env.regularCubes(args)
    env.randomCubes(args)
    rm = roboticMoving(args)

    # start stack
    state_durations=[0.5, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3] # the simulate time in every motion
    pb.setTimeStep=args.control_dt
    colnum = int(math.sqrt(args.cubeNum))
    for i in range(colnum):
        for j in range(colnum-1):
            state_t=0.
            current_state=0
            while True:
                state_t+=args.control_dt
                pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
                
                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                        width=320, height=240,
                        viewMatrix=env.viewList,
                        projectionMatrix=env.projectionList)

                xpos = args.cubeBasePosition[0]
                ypos = args.cubeBasePosition[1]
                # zpos = args.cubeBasePosition[2]
                if current_state == 0:
                    rm.robotPosInit(args.robotInitAngle)

                if current_state == 1:
                    rm.setpos(6, [xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])
                    
                if current_state == 2:
                    rm.setpos(6, [xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.25], [0, math.pi, -math.pi/4])

                if current_state == 3:
                    rm.gripperPick()

                if current_state == 4:
                    rm.setpos(6, [xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])

                if current_state == 5:
                    rm.setpos(6, [xpos, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])
                
                if current_state == 6:
                    rm.gripperPush()

                if state_t>state_durations[current_state]:
                    if current_state == 0:
                        pass
                        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                        width=args.cameraImgWidth, height=args.cameraImgHight,
                        viewMatrix=env.viewList,
                        projectionMatrix=env.projectionList)
                        # lightDirection=[0, 0, 1],
                        # lightColor=[1, 1, 1],
                        # lightDistance=1.,
                        # renderer = pb.ER_BULLET_HARDWARE_OPENGL)
                        depthImgReal = transDepthBufferToRealZ(width, height, depthImg, env.projectionMatrix)

                        # cv2.imwrite("img/framecalibrateRGB.png", rgbImg) 
                        # cv2.imwrite("img/framecalibrateDepth.exr", depthImgReal)
                        # env.cameraTakePhoto(args, i, j)
                    current_state+=1
                    if current_state>=len(state_durations):
                        break
                    state_t=0

                pb.stepSimulation()
