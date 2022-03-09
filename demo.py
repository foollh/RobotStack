import time
import math
from turtle import width
import numpy as np
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving

def setCameraPicAndGetPic(robot_id : int, width : int = 224, height : int = 224, physicsClientId : int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    basePos, baseOrientation = pb.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = pb.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])              # 变换后的x轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])              # 变换后的z轴

    basePos = np.array(basePos)
    BASE_RADIUS = 1.
    BASE_THICKNESS = 1.
    # 摄像头的位置
    cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
    targetPos = cameraPos + 1 * tx_vec

    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )
    projectionMatrix = pb.computeProjectionMatrixFOV(
        fov=50.0,               # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,            # 摄像头焦距下限
        farVal=20,               # 摄像头能看上限
        physicsClientId=physicsClientId
    )
    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )
    
    return width, height, rgbImg, depthImg, segImg

if __name__ == "__main__":
    args = Args()
    env = robotEnvironment(args)
    env.basic_env(args)

    env.regularCubes(args)
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

                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=224, height=224,
                    viewMatrix=env.viewMatrix,
                    projectionMatrix=env.projectMatrix
                )

                state_t+=args.control_dt
                pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)

                xpos = args.cubeBasePosition[0]
                ypos = args.cubeBasePosition[1]
                # zpos = args.cubeBasePosition[2]
                if current_state == 0:
                    rm.robotPosInit(args.robotInitAngle)

                if current_state == 1:
                    rm.setpos([xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])

                if current_state == 2:
                    rm.setpos([xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.25], [0, math.pi, -math.pi/4])

                if current_state == 3:
                    rm.gripperPick()

                if current_state == 4:
                    rm.setpos([xpos + (2-j)*args.cubeInterval, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])

                if current_state == 5:
                    rm.setpos([xpos, 0.1*(i-1), 0.4], [0, math.pi, -math.pi/4])
                
                if current_state == 6:
                    rm.gripperPush()

                if state_t>state_durations[current_state]:
                    current_state+=1
                    if current_state>=len(state_durations):
                        break
                    state_t=0

                pb.stepSimulation()