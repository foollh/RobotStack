import numpy as np
import cv2

#定义形状检测函数
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

    keypointArray = np.array(keyPoints).reshape(len(keyPoints), -1)
    # print("keyPoints\n", keypointArray)
    return keypointArray
    

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
    # cv2.imwrite("img/Original_img.png", img)
    # cv2.imwrite("img/imgGray.png", imgGray)
    # cv2.imwrite("img/imgBlur.png", imgBlur)
    cv2.imwrite("img/imgCanny.png", imgCanny)
    cv2.imwrite("img/shape_Detection.png", imgContour)

    return keypointArray

def getKeypointsDepth(imagePoints):
    imagePointsDepth = []
    imgDepth = cv2.imread("img/framecalibrateDepth.exr", -1)
    # print("imgDepth.shape\n", imgDepth.shape)
    for point in imagePoints:
        imagePointsDepth.append(imgDepth[int(point[0]), int(point[1])])
    return np.array(imagePointsDepth)

def cubeDetection(image):
    '''
    inputs: image->numpy.array
    outputs: drewImage, Index, Keypoint, radius, orientation
    '''
    if image.ndim == 3:
        imgGray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  #转灰度图
    else:
        imgGray = image.copy()
    
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 100, 7)

    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    keyPoints = []
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        cv2.drawContours(image, obj, -1, (255, 0, 0), 2)  #绘制轮廓线
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度
        if area > 50 and area < 1000:
            keyPoints.append([x+w/2, y+h/2])
            cv2.circle(image, (int(x+w/2), int(y+h/2)), 1, (255, 0, 0))
            cv2.putText(image, str(len(keyPoints)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)  #绘制边界框

    # cv2.imwrite("img/imgBinary.png", dst)
    # cv2.imwrite("img/Original_img.png", img)
    cv2.imwrite("img/imgGray.png", imgGray)
    cv2.imwrite("img/imgBlur.png", imgBlur)
    cv2.imwrite("img/imgCanny.png", imgCanny)
    cv2.imwrite("img/shape_Detection.png", image)


if __name__ == "__main__":
    imgPath = "img/framecalibrateRGB.png"
    img = cv2.imread(imgPath)
    cubeDetection(img)
