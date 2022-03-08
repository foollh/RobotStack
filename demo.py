import time
import math
import numpy as np
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving

args = Args()
env = robotEnvironment(args)

env.basic_env(args)
env.regularCubes(args)

# if args.useSimulation and args.useRealTimeSimulation == 1:
#     pb.setRealTimeSimulation(args.useRealTimeSimulation)

# if (args.useSimulation and args.useRealTimeSimulation == 0):
state_durations=[1, 0.5, 0.5, 0.5, 0.5, 0.5, 1,0.5, 0.5, 0.5, 0.5, 1]
state_t=0.
current_state=0

pb.setTimeStep=args.control_dt

rm = roboticMoving(args)

while True:
    state_t+=args.control_dt
    pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
    if current_state == 0:
        rm.robotPosInit(args.robotInitAngle)

    if current_state == 1:
        rm.setpos([0.75, -0.1, 0.25], [0, math.pi, -math.pi/4])

    if current_state == 2:
        rm.gripperPick()

    if current_state == 3:
        rm.getpos()
        rm.setpos([0.75, -0.1, 0.4], [0, math.pi, -math.pi/4])

    if current_state == 4:
        rm.getpos()
        rm.setpos([0.55, -0.1, 0.4], [0, math.pi, -math.pi/4])
    
    if current_state == 5:
        rm.gripperPush()

    if current_state == 6:
        rm.setpos([0.65, -0.1, 0.25], [0, math.pi, -math.pi/4])

    if current_state == 7:
        rm.gripperPick()

    if current_state == 8:
        rm.getpos()
        rm.setpos([0.65, -0.1, 0.4], [0, math.pi, -math.pi/4])

    if current_state == 9:
        rm.getpos()
        rm.setpos([0.55, -0.1, 0.4], [0, math.pi, -math.pi/4])
    
    if current_state == 10:
        rm.gripperPush()

    if state_t>state_durations[current_state]:
        current_state+=1
        if current_state>=len(state_durations):
            current_state=0
        state_t=0

    pb.stepSimulation()

