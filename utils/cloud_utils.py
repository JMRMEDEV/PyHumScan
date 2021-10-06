import numpy as np

def denormalize_RGB(color_data):
    """
    Take all the colors from a cloud for converting
    from 0-1 normalized RGB to 0-255 (non-normalized)
    for implementing colored-joint detection algorithm
    """
    for row in range(color_data.shape[0]):
        for col in range(color_data.shape[1]):
            color_data[row][col] = int(round((color_data[row][col] * 255) / 1))
    return color_data

def get_joint_centre(joint_cloud):
    """
    Calculates the average from the points in a given
    joint for determing its approximated centre.
    """
    xs = []
    ys = []
    zs = []
    x = 0
    y = 0
    z = 0

    for row in range(joint_cloud.shape[0]):
        xs.append(joint_cloud[row][0])
        ys.append(joint_cloud[row][1])
        zs.append(joint_cloud[row][2])
    
    for i in range(len(xs)):
        x = x + xs[i]

    for i in range(len(ys)):
        y = y + ys[i]

    for i in range(len(zs)):
        z = z + zs[i]

    if len(xs) < 1 or len(ys) < 1 or len(zs) < 1:
        return []

    joint_centre = [x, y, z]

    return joint_centre

def get_joints_points(point_cloud):
    """
    Remove all the points that are not joints, create an
    output cloud with the small created clouds that did contain
    the joints
    """
    pose3d = {
        'forehead': None,
        'chin': None,
        'shoulder1': None,
        'shoulder2': None,
        'elbow1': None,
        'elbow2': None,
        'wrist1': None,
        'wrist2': None,
        'hip1': None,
        'hip2': None,
        'ankle1': None,
        'ankle2': None
    }

    # Remove all non-joint points
    forehead = point_cloud[np.all(point_cloud[:,3:6] == np.array([250, 0, 0]), axis=1)]
    forehead = get_joint_centre(forehead)
    chin = point_cloud[np.all(point_cloud[:,3:6] == np.array([250, 250, 0]), axis=1)]
    chin = get_joint_centre(chin)
    shoulder1 = point_cloud[np.all(point_cloud[:,3:6] == np.array([0, 250, 0]), axis=1)]
    shoulder1 = get_joint_centre(shoulder1)
    shoulder2 = point_cloud[np.all(point_cloud[:,3:6] == np.array([250, 112, 0]), axis=1)]
    shoulder2 = get_joint_centre(shoulder2)
    elbow1 = point_cloud[np.all(point_cloud[:,3:6] == np.array([250, 0, 250]), axis=1)]
    elbow1 = get_joint_centre(elbow1)
    elbow2 = point_cloud[np.all(point_cloud[:,3:6] == np.array([0, 250, 200]), axis=1)]
    elbow2 = get_joint_centre(elbow2)
    wrist1 = point_cloud[np.all(point_cloud[:,3:6] == np.array([158, 250, 0]), axis=1)]
    wrist1 = get_joint_centre(wrist1)
    wrist2 = point_cloud[np.all(point_cloud[:,3:6] == np.array([0, 112, 250]), axis=1)]
    wrist2 = get_joint_centre(wrist2)
    hip1 = point_cloud[np.all(point_cloud[:,3:6] == np.array([170, 0, 255]), axis=1)]
    hip1 = get_joint_centre(hip1)
    hip2 = point_cloud[np.all(point_cloud[:,3:6] == np.array([190, 248, 255]), axis=1)]
    hip2 = get_joint_centre(hip2)
    ankle1 = point_cloud[np.all(point_cloud[:,3:6] == np.array([172, 200, 0]), axis=1)]
    ankle1 = get_joint_centre(ankle1)
    ankle2 = point_cloud[np.all(point_cloud[:,3:6] == np.array([156, 82, 220]), axis=1)]
    ankle2 = get_joint_centre(ankle2)

    output_cloud = np.array([
        forehead,
        chin,
        shoulder1,
        shoulder2,
        elbow1,
        elbow2, 
        wrist1,
        wrist2,
        hip1,
        hip2,
        ankle1,
        ankle2
    ])

    joints_to_delete = []

    for i in range(len(output_cloud)):
        if len(output_cloud[i]) < 1:
                joints_to_delete.append(i)

    output_cloud = np.delete(output_cloud, joints_to_delete, 0)

    pose3d['forehead'] = forehead
    pose3d['chin'] = chin
    pose3d['shoulder1'] = shoulder1
    pose3d['shoulder2'] = shoulder2
    pose3d['elbow1'] = elbow1
    pose3d['elbow2'] = elbow2
    pose3d['wrist1'] = wrist1
    pose3d['wrist2'] = wrist2
    pose3d['hip1'] = hip1
    pose3d['hip2'] = hip2
    pose3d['ankle1'] = ankle1
    pose3d['ankle2'] = ankle2

    return pose3d, output_cloud