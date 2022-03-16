import numpy as np

def getCameraPoseFromViewMatrix(viewMatrix):
    # get camera pose from the view matrix
    view_array = np.array(viewMatrix).reshape(4, 4)
    
    cameraPose = np.eye(4)
    N = view_array[:3, 2].reshape(3, 1)
    U = view_array[:3, 0].reshape(3, 1)
    V = view_array[:3, 1].reshape(3, 1)
    R = np.concatenate((U, V, N), axis=1)
    T =  - R @ view_array[-1, :3]
    cameraPose[:3, :3] = R
    cameraPose[:3, -1] = T
    # N = np.array(args.cameraPos - args.cameraFocus)
    # U = np.cross(np.array(args.cameraVector), N)
    # V = np.cross(N, U)
    return cameraPose
        