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