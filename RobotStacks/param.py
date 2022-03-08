import math

class Args():
    def __init__(self) -> None:
        # environment param
        self.GUI = True
        self.gravity = -10
        self.visualShapeId = -1
        self.useSimulation = 1
        self.useRealTimeSimulation = 0
        self.control_dt = 1./240.

        # robot param
        # self.robotType = "SDF"  # SDF or URDF
        # self.robotPath = "kuka_iiwa/kuka_with_gripper.sdf"
        self.robotType = "URDF"  # SDF or URDF
        self.robotPath = "franka_panda/panda.urdf"
        
        self.robotEndEffectorIndex = 6
        self.robotScale = 1
        self.robotInitAngle = [0.0, math.pi/4.-0.3, 0.0, -math.pi/2. + 0.3, 0.0, 3*math.pi/4., -math.pi/4., -math.pi/2., -math.pi/2., 1, 1, 0]
        

        # other UDRF param
        self.tablePath = "table/table.urdf"
        self.tablePosition = [0.5,0,-0.65]
        self.trayPath = "tray/traybox.urdf"
        self.trayPosition = [0.65,0,0]
        
        self.cubeNum = 9 
        self.cubeRadius = 0.025
        self.cubeMass = 0.4
        self.cubeInterval = 0.1
        self.cubeBasePosition = [0.55, -0.1, 0.1]
        self.cubeBaseOrientation = [0, 0, 0]

        
