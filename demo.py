from ast import arg
import time
import math
import numpy as np
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving

if __name__ == "__main__":
    args = Args()
    env = robotEnvironment(args)

    env.basic_env(args)
    env.regularCubes(args)

    state_durations=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    pb.setTimeStep=args.control_dt

    rm = roboticMoving(args)

    colnum = int(math.sqrt(args.cubeNum))
    for i in range(colnum):
        for j in range(colnum-1):
            state_t=0.
            current_state=0
            while True:
                state_t+=args.control_dt
                pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)

                xpos = args.cubeBasePosition[0]
                ypos = args.cubeBasePosition[1]
                # zpos = args.cubeBasePosition[2]
                if current_state == 0:
                    rm.robotPosInit(args.robotInitAngle)

                if current_state == 1:
                    rm.setpos([xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.25], [0, math.pi, -math.pi/4])

                if current_state == 2:
                    rm.gripperPick()

                if current_state == 3:
                    rm.setpos([xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])

                if current_state == 4:
                    rm.setpos([xpos, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])
                
                if current_state == 5:
                    rm.gripperPush()

                if state_t>state_durations[current_state]:
                    current_state+=1
                    if current_state>=len(state_durations):
                        break
                    state_t=0

                pb.stepSimulation()