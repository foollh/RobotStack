import time
import math
import numpy as np
import cv2
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving


if __name__ == "__main__":
    args = Args()
    env = robotEnvironment(args)
    env.basic_env(args)

    # env.regularCubes(args)
    env.randomCubes(args)
    rm = roboticMoving(args)

    # start stack
    state_durations=[0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3] # the simulate time in every motion
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
                        viewMatrix=env.viewMatrix,
                        projectionMatrix=env.projectionMatrix)

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
                        # env.cameraTakePhoto(args, i, j)
                    current_state+=1
                    if current_state>=len(state_durations):
                        break
                    state_t=0

                pb.stepSimulation()
