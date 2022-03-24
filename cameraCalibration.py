import cv2
import numpy as np
import time
import math
import pybullet as pb
from RobotStacks import Args, robotEnvironment, roboticMoving, transDepthBufferToRealZ

def ShapeDetection(img, imgContour):
    keyPoints = []
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 1)  #绘制轮廓线
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度

        if area > 50 and area < 1000:
            # print("area:\n", area)
            # print("CornerNum:\n", CornerNum)
            # print("approx:\n", approx)
            # print("x, y, w, h\n", [x, y, w, h])
            keyPoints.append([x+w/2, y+h/2])
            cv2.circle(imgContour, (int(x+w/2), int(y+h/2)), 1, (255, 0, 0))
            cv2.putText(imgContour, str(len(keyPoints)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)  #绘制边界框

    if keyPoints == []:
        return np.array([[0, 0], [0, 0], [0, 0]])
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

    path = 'img/framecalibrateRGB.png'
    # path = 'testPandaRobot.png'
    img = cv2.imread(path)
    imgContour = img.copy()
    keyPoints = []

    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  #转灰度图
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)  #高斯模糊

    ret, dst= cv2.threshold(imgGray, thresh=230, maxval=255, type=cv2.THRESH_BINARY)
    imgCanny = cv2.Canny(dst,60,60)  #Canny算子边缘检测
    keypointArray = ShapeDetection(imgCanny, imgContour)  #形状检测

    cv2.imwrite("img/imgBinary.png", dst)
    cv2.imwrite("img/imgGray.png", imgGray)
    # cv2.imwrite("img/imgBlur.png", imgBlur)
    cv2.imwrite("img/imgCanny.png", imgCanny)
    cv2.imwrite("img/shape_Detection.png", imgContour)

    return keypointArray

def getKeypointsDepth(imagePoints):
    imagePointsDepth = []
    imgDepth = cv2.imread("img/framecalibrateDepth.exr", cv2.IMREAD_UNCHANGED)
    # print("imgDepth.shape\n", imgDepth.shape)
    for point in imagePoints:
        imagePointsDepth.append(imgDepth[int(point[1]), int(point[0])])
    return np.array(imagePointsDepth)

def main():

    args = Args()

    # camera param  0.8: 0.8753687  0.6: 0.8302822  0.5:0.7935055
    args.cameraPos = [0.65, -0.1, 0.8]
    args.cameraFocus = [0.65, 0., 0.]
    args.cameraVector = [1., 0., 0.]
    args.cameraFov = 90
    args.cameraAspect = 640/480
    args.cameraNearVal = 0.01
    args.cameraFarVal = 10

    env = robotEnvironment(args)
    # print("camera intrinsics matrix:\n", env.intrinsicsMatrix)
    # print("view matrix:\n", env.viewMatrix)
    
    # temp = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    env.basic_env(args)

    cubeCenterPos = env.regularCubes(args)
    # print("cubeCenterPos\n", cubeCenterPos.shape)

    # env.randomCubes(args)
    # rm = roboticMoving(args)
    # print("base position:\n", pb.getBasePositionAndOrientation(rm.robotId))
    
    state_durations=[0.3, 0.2, 0.2]  # the simulate time in every motion
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
                projectionMatrix=env.projectionList)

        if state_t>state_durations[current_state]:
            if current_state == 0:
                # pb.removeBody(rm.robotId)
                # [0.55, -0.1, 0.1]

                # drawPoints([0.55, -0.1, 0.025], debugLineLen, color=[1, 0, 0])
                # drawPoints([0.65, 0., 0.025], debugLineLen, color=[1, 0, 0])

                width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=args.cameraImgWidth, height=args.cameraImgHight,
                    viewMatrix=env.viewList,
                    projectionMatrix=env.projectionList)
                    # lightDirection=[0, 0, 1],
                    # lightColor=[1, 1, 1],
                    # lightDistance=1.,
                    # renderer = pb.ER_BULLET_HARDWARE_OPENGL)
                depthImgReal = transDepthBufferToRealZ(width, height, depthImg, env.projectionMatrix)

                cv2.imwrite("img/framecalibrateRGB.png", rgbImg) 
                cv2.imwrite("img/framecalibrateDepth.exr", depthImgReal)

            current_state+=1
            if current_state>=len(state_durations):
                break
                current_state = 2
            state_t=0

        pb.stepSimulation()
    
    imagePoints = calibrateCubes()
    # print("imageKeypoints:\n", imagePoints)
    imagePointsDepth = getKeypointsDepth(imagePoints)
    # print("imageKeypointsDepth:\n", imagePointsDepth)

    _, R, T = cv2.solvePnP(cubeCenterPos, imagePoints, env.intrinsicsMatrix, env.distCoeffs)

    RT = np.eye(4)
    RT[:3, :3] = cv2.Rodrigues(R)[0]
    RT[:3, -1] = T.reshape(3)   
    print("RT transform:\n", RT)
    print("RT inv transform:\n", np.linalg.inv(RT))
    print("view Matrix:\n", env.viewMatrix)
    print("inv view matrix:\n", np.linalg.inv(env.viewMatrix))
    print("extrinsics matrix:\n", env.extrinsicMatrix)  
    print("inv extrinsics matrix:\n", np.linalg.inv(env.extrinsicMatrix))  

    homoPoint = np.ones(3)
    cameraPoint = np.ones([4, 1])
    homoPoint[:2] = imagePoints[4]
    homoPoint = imagePointsDepth[4] * homoPoint

    cameraPoint[:3] = np.linalg.inv(env.intrinsicsMatrix) @ homoPoint.reshape(3, 1)

    worldPoint = np.array([[0.65], [0.], [0.025], [1.]])
    worldPointInfer = np.linalg.inv(RT) @ cameraPoint
    
    print("centerWorldPointInfer:\n", worldPointInfer)
    print("centerWorldPointReal:\n", worldPoint)

if __name__ == "__main__":
    main()
