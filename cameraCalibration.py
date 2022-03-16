import cv2
import numpy as np
import time
import math
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving, getCameraPoseFromViewMatrix

#定义形状检测函数
def ShapeDetection(img, imgContour):
    keyPoints = []
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        # cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 1)  #绘制轮廓线
        # if area < 2:
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度

        if area > 60 and area < 1000:
            # print("area:\n", area)
            # print("CornerNum:\n", CornerNum)
            # print("approx:\n", approx)
            # print("x, y, w, h\n", [x, y, w, h])
            keyPoints.append([x+w/2, y+h/2])
            cv2.circle(imgContour, (int(x+w/2), int(y+h/2)), 1, (255, 0, 0))
            cv2.putText(imgContour, str(len(keyPoints)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)  #绘制边界框

    keypointArray = np.array(keyPoints).reshape(len(keyPoints), -1)
    # print("keyPoints\n", keypointArray)
    return keypointArray
    

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

def calibrateCubes():
    # img = cv2.imread('testPandaRobot.png', cv2.IMREAD_UNCHANGED)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.findContours()

    path = 'testPandaRobot.png'
    img = cv2.imread(path)
    imgContour = img.copy()
    keyPoints = []

    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  #转灰度图
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)  #高斯模糊

    ret, dst= cv2.threshold(imgGray, thresh=230, maxval=255, type=cv2.THRESH_BINARY)
    imgCanny = cv2.Canny(dst,60,60)  #Canny算子边缘检测
    keypointArray = ShapeDetection(imgCanny, imgContour)  #形状检测

    cv2.imwrite("img/imgBinary.png", dst)
    cv2.imwrite("img/Original_img.png", img)
    cv2.imwrite("img/imgGray.png", imgGray)
    cv2.imwrite("img/imgBlur.png", imgBlur)
    cv2.imwrite("img/imgCanny.png", imgCanny)
    cv2.imwrite("img/shape_Detection.png", imgContour)

    return keypointArray

def main():

    args = Args()

    # camera param
    args.cameraPos = [0.5, 0., 0.8]
    args.cameraFocus = [0.6, 0., 0.]
    args.cameraVector = [1., 0., 0.]
    args.cameraFov = 90
    args.cameraAspect = 640/480
    args.cameraNearVal = 0.1
    args.cameraFarVal = 20

    env = robotEnvironment(args)
    # print("camera intrinsics matrix:\n", env.intrinsicsMatrix)
    viewMatrix_reshape = np.array(env.viewMatrix, np.float32).reshape(4, 4)
    print("viewMatrix_reshape:\n", viewMatrix_reshape)

    # temp = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # print("view matrix opencv:\n", viewMatrix_reshape @ temp)
    env.basic_env(args)

    cubeCenterPos = env.regularCubes(args)
    # print("cubeCenterPos\n", cubeCenterPos.shape)

    # env.randomCubes(args)
    # rm = roboticMoving(args)
    # print("base position:\n", pb.getBasePositionAndOrientation(rm.robotId))
    
    state_durations=[0.5, 0.2, 0.2]  # the simulate time in every motion
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

        if state_t>state_durations[current_state]:
            if current_state == 0:
                # pb.removeBody(rm.robotId)
                # [0.55, -0.1, 0.1]

                # drawPoints([0.55, -0.1, 0.025], debugLineLen, color=[1, 0, 0])
                # drawPoints([0.65, -0., 0.025], debugLineLen, color=[1, 0, 0])

                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=args.cameraImgWidth, height=args.cameraImgHight,
                    viewMatrix=env.viewMatrix,
                    projectionMatrix=env.projectionMatrix,
                    renderer = pb.ER_TINY_RENDERER)
                
                cv2.imwrite("testPandaRobot.png", rgbImg) 

            current_state+=1
            if current_state>=len(state_durations):
                break
                current_state = 2
            state_t=0

        pb.stepSimulation()
    
    return env.intrinsicsMatrix, cubeCenterPos

if __name__ == "__main__":
    cameraMatrix, objectPoints = main()
    distCoeffs = np.mat([0,0,0,0,0])
    imagePoints = calibrateCubes()

    _, R, T = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
    print('所求结果:')
    print("旋转向量",R)
    print("旋转矩阵:", cv2.Rodrigues(R)[0])
    print("平移向量",T)
    flip_axis = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R_pose = (cv2.Rodrigues(R)[0] @ flip_axis).T
    camera_view_matrix = np.eye(4)
    camera_view_matrix[:3, :3] = R_pose
    camera_view_matrix[-1, :3] = -T.reshape(3,)
    print("camera_view_matrix:\n", camera_view_matrix)

    print("camera_pose:\n", getCameraPoseFromViewMatrix(camera_view_matrix))
    
    # print("欧拉角：\n", rotationMatrixToEulerAngles(cv2.Rodrigues(R)[0]))
    