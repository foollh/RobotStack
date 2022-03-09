import sys
import math
from tkinter.messagebox import NO
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

    def getpos(self):
        pass

    def setpos(self, position, orientation):
        pos = position
        orn = pb.getQuaternionFromEuler(orientation)
        # jointPoses = self.accurateCalculateInverseKinematics(pos, orn, threshold=0.001, maxIter=100)
        jointPoses = pb.calculateInverseKinematics(self.robotId, self.endEffectorIndex, pos, orn)
        for j in range(self.numJoints-5):
            pb.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=j,
                                controlMode=pb.POSITION_CONTROL,
                                targetPosition=jointPoses[j])


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
    