import math
import copy
import pybullet as pb
import pybullet_data as pd

class robotEnvironment():
    def __init__(self, args):
        if args.GUI:
            pb.connect(pb.GUI)
    
    def basic_env(self, args):
        pb.setGravity(0, 0, args.gravity)
        pb.setAdditionalSearchPath(pd.getDataPath())
        
        pb.loadURDF("plane.urdf", [0, 0, -0.3])
        pb.loadURDF(args.tablePath, basePosition=args.tablePosition)
        pb.loadURDF(args.trayPath, basePosition=args.trayPosition)

    def regularCubes(self, args):
        cubeUid = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[args.cubeRadius, args.cubeRadius, args.cubeRadius])
        colnum = int(math.sqrt(args.cubeNum))
        
        for i in range(colnum):
            for j in range(colnum):
                baseposition = copy.deepcopy(args.cubeBasePosition)
                baseposition[0] = args.cubeBasePosition[0] + i * args.cubeInterval
                baseposition[1] = args.cubeBasePosition[1] + j * args.cubeInterval
                baseorientation = pb.getQuaternionFromEuler(args.cubeBaseOrientation)
                pb.createMultiBody(args.cubeMass, cubeUid, args.visualShapeId, baseposition, baseorientation)

    def randomCubes(self, args):
        radius = 0.025
        cubeUid = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[radius, radius, radius])

        # sphereRadius = 0.025
        # colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        # colBoxId = p.createCollisionShape(p.GEOM_BOX,
        #                                   halfExtents=[sphereRadius, sphereRadius, sphereRadius])
        # mass = 1
        # visualShapeId = -1
        # for i in range(3):
        #   for j in range(3):
        #     for k in range(10):
        #         #随即方块的位置
        #         xpos = 0.2 + 0.1*i + 0.5*random.random()  
        #         ypos = -0.35 + 0.1*j + 0.5* random.random()  
        #         zpos = 0.1 + 0.1*k + 0.5*random.random()
        #         ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        #         baseOrientation = p.getQuaternionFromEuler([0, 0, ang])
        #         basePosition = [xpos, ypos, zpos]
        #         if (k % 2):
        #             sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, basePosition,
        #                                         baseOrientation)
        #         else:
        #             sphereUid = p.createMultiBody(mass,
        #                                         colBoxId,
        #                                         visualShapeId,
        #                                         basePosition,
        #                                         baseOrientation)
