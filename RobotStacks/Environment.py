import math
import copy
import cv2
import random
import numpy as np
import pybullet as pb
import pybullet_data as pd

class robotEnvironment():
    def __init__(self, args):
        if args.GUI:
            pb.connect(pb.GUI)

        # set openGL camera param
        self.viewList = pb.computeViewMatrix(args.cameraPos, args.cameraFocus, args.cameraVector)
        self.viewMatrix = np.asarray(self.viewList).reshape([4,4],order='F')
        self.projectionList = pb.computeProjectionMatrixFOV(args.cameraFov, args.cameraAspect, args.cameraNearVal, args.cameraFarVal)
        self.projectionMatrix = np.asarray(self.projectionList).reshape([4,4],order='F')

        # get camera intrinsics and extrinsics matrix from camera settings
        self.intrinsicsMatrix = np.eye(3)
        self.intrinsicsMatrix[0, 0] = args.cameraImgWidth * self.projectionList[0] / 2
        self.intrinsicsMatrix[1, 1] = args.cameraImgHight * self.projectionList[5] / 2
        self.intrinsicsMatrix[0, 2] = args.cameraImgWidth * (1 - self.projectionList[8]) / 2
        self.intrinsicsMatrix[1, 2] = args.cameraImgHight * (1 + self.projectionList[9]) / 2
        self.distCoeffs = np.mat([0,0,0,0,0])

        flip_axis = np.array([[1., 0, 0, 0], 
                        [0, -1., 0, 0], 
                        [0, 0, -1., 0],
                        [0, 0, 0, 1.]])
        self.extrinsicMatrix = self.viewMatrix @ flip_axis

    def basic_env(self, args):
        pb.setGravity(0, 0, args.gravity)
        pb.setAdditionalSearchPath(pd.getDataPath())
        
        # load urdf models
        pb.loadURDF("plane.urdf", [0, 0, -0.3])
        pb.loadURDF(args.tablePath, basePosition=args.tablePosition)
        pb.loadURDF(args.trayPath, basePosition=args.trayPosition)

    def cameraTakePhoto(self, args, i, j):
        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                    width=args.cameraImgWidth, height=args.cameraImgHight,
                    viewMatrix=self.viewList,
                    projectionMatrix=self.projectionList,
                    renderer = pb.ER_TINY_RENDERER)
        cv2.imwrite(args.rgbdPath + "frame_rgb" + str(i) + str(j) + ".png", rgbImg)
        cv2.imwrite(args.rgbdPath + "frame_depth" + str(i) + str(j) + ".tiff", depthImg)
        # cv2.imwrite(rgbdPath + "frame_seg" + str(i) + str(j) + ".png", segImg)

    def regularCubes(self, args):
        cubeCenterPos = []
        cubeUid = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[args.cubeRadius, args.cubeRadius, args.cubeRadius])
        colnum = int(math.sqrt(args.cubeNum))
        
        for i in range(colnum):
            for j in range(colnum):
                baseposition = copy.deepcopy(args.cubeBasePosition)
                baseposition[0] = args.cubeBasePosition[0] + i * args.cubeInterval
                baseposition[1] = args.cubeBasePosition[1] + j * args.cubeInterval
                cubeCenterPos.append([baseposition[0], baseposition[1], args.cubeRadius])
                baseorientation = pb.getQuaternionFromEuler(args.cubeBaseOrientation)
                pb.createMultiBody(args.cubeMass, cubeUid, args.visualShapeId, baseposition, baseorientation)
        
        return np.array(cubeCenterPos)  # the center point position of cubes

    def randomCubes(self, args):
        cubeUid = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[args.cubeRadius, args.cubeRadius, args.cubeRadius])
        layerNum = int((args.cubeNum)**(1/3))
        visualShapeId = -1
        for i in range(layerNum):
          for j in range(layerNum):
            for k in range(layerNum):
                #随即方块的位置
                xpos = args.trayPosition[0] + 5*args.cubeRadius*(i - int(layerNum/2) + random.random() - 0.5)
                ypos = args.trayPosition[1] + 5*args.cubeRadius*(j - int(layerNum/2) + random.random() - 0.5)
                zpos = args.trayPosition[2] + 5*args.cubeRadius*(k+1+random.random())
                ang = 3.14 * 0.5 + 3.1415925438 * random.random()
                baseOrientation = pb.getQuaternionFromEuler([0, 0, ang])
                basePosition = [xpos, ypos, zpos]
                sphereUid = pb.createMultiBody(args.cubeMass,
                                            cubeUid,
                                            visualShapeId,
                                            basePosition,
                                            baseOrientation)
