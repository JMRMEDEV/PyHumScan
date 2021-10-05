def get_joint_color(joint_name):
    """
    Returns a BGR color for a given joint
    """
    if joint_name == "forehead":
        return (0, 0, 250)
    if joint_name == "chin":
        return (0, 250, 250)
    if joint_name == "shoulder1":
        return (0, 250, 0)
    if joint_name == "shoulder2":
        return (0, 112, 250)
    if joint_name == "elbow1":
        return (250, 0, 250)
    if joint_name == "elbow2":
        return (200, 250, 0)
    if joint_name == "wrist1":
        return (0, 250, 158)
    if joint_name == "wrist2":
        return (250, 112, 0)
    if joint_name == "hip1":
        return (255, 0, 170)
    if joint_name == "hip2":
        return (255, 248, 190)
    if joint_name == "ankle1":
        return (0, 200, 172)
    if joint_name == "ankle2":
        return (220, 82, 156)

def get_defined_joint():
    """
    Returns a numpy array with the joint to track
    """
