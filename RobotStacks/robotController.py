import sys
import math
import random
import pybullet as pb

class roboticMoving():
    def __init__(self, args):
        if args.robotType == "SDF":
            self.robotId = (pb.loadSDF(args.robotPath))[0]
        elif args.robotType == "URDF":
            self.robotId = pb.loadURDF(args.robotPath, useFixedBase=True)
        else:
            print("The robot type can not be read!")
            sys.exit()
        self.numJoints = pb.getNumJoints(self.robotId)
        self.endEffectorIndex = args.robotEndEffectorIndex
        self.gripper = args.robotGrippers

    def robotPosInit(self, angleList):
        for idx in range(self.numJoints):
            pb.setJointMotorControl2(self.robotId, idx, pb.POSITION_CONTROL, angleList[idx])
    
    def drawDebugPoint(self, idx, debugLineLen, color):
        posIndex = self.getpos(idx)
        pre_pos1 = [posIndex[0][0]-debugLineLen, posIndex[0][1], posIndex[0][2]]
        tar_pos1 = [posIndex[0][0]+debugLineLen, posIndex[0][1], posIndex[0][2]]
        pre_pos2 = [posIndex[0][0], posIndex[0][1]-debugLineLen, posIndex[0][2]]
        tar_pos2 = [posIndex[0][0], posIndex[0][1]+debugLineLen, posIndex[0][2]]
        pre_pos3 = [posIndex[0][0], posIndex[0][1], posIndex[0][2]-debugLineLen]
        tar_pos3 = [posIndex[0][0], posIndex[0][1], posIndex[0][2]+debugLineLen]

        pb.addUserDebugLine(pre_pos1, tar_pos1,lineColorRGB=color, lineWidth=300)
        pb.addUserDebugLine(pre_pos2, tar_pos2,lineColorRGB=color, lineWidth=300)
        pb.addUserDebugLine(pre_pos3, tar_pos3,lineColorRGB=color, lineWidth=300)
        pb.addUserDebugText(str(idx), pre_pos1, textColorRGB=[1, 0, 0], textSize=2.)
        # print("link{0} position:\n{1}".format(idx, posIndex))
    
    def getpos(self, linkIndex):
        linkInfo = pb.getLinkState(self.robotId, linkIndex)
        linkToBasePos, linkToBaseOrn = linkInfo[0], linkInfo[1]
        worldPosition, worldOrientation = linkInfo[4], linkInfo[5]
        return worldPosition, worldOrientation

    def setpos(self, jointIndex, position, orientation):
        pos = position
        orn = orientation
        if len(orientation) == 3:
            orn = pb.getQuaternionFromEuler(orientation)
        # jointPoses = self.accurateCalculateInverseKinematics(pos, orn, threshold=0.001, maxIter=100)
        jointPoses = pb.calculateInverseKinematics(self.robotId, jointIndex, pos, orn)
        for j in range(self.numJoints-5):
            pb.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=j,
                                controlMode=pb.POSITION_CONTROL,
                                targetPosition=jointPoses[j],
                                maxVelocity=5.)

    def accurateCalculateInverseKinematics(self, targetPos, targetOrn, threshold, maxIter):
        closeEnough = False
        iter = 0
        dist2 = 1e30
        while (not closeEnough and iter < maxIter):
            jointPoses = pb.calculateInverseKinematics(self.robotId, self.endEffectorIndex, targetPos, targetOrn)
            for i in range(self.numJoints-5):
                pb.resetJointState(self.robotId, i, jointPoses[i])
            ls = pb.getLinkState(self.robotId, self.endEffectorIndex)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1
            pb.stepSimulation()
        #print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
        return jointPoses

    def gripperPick(self):
        pb.setJointMotorControl2(self.robotId, self.gripper[0], pb.POSITION_CONTROL,force=500)
        pb.setJointMotorControl2(self.robotId, self.gripper[1], pb.POSITION_CONTROL,force=500)
    
    def gripperPush(self):
        pb.setJointMotorControl2(self.robotId, self.gripper[0], pb.POSITION_CONTROL,1)
        pb.setJointMotorControl2(self.robotId, self.gripper[1], pb.POSITION_CONTROL,1)
