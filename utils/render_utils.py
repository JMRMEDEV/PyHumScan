import io

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image


def render_3d(joints_cloud):
    """
    Takes an input point cloud in format of list and
    render the joint locations
    """
    # Extract the corresponding columns
    x = [row[0] for row in joints_cloud]
    y = [row[1] for row in joints_cloud]
    z = [row[2] for row in joints_cloud]
    # Load into a plotly figure the xs, ys, and sz and show them as
    # markers (spheres)
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode="markers")])
    # Set the plotly camera according to Kinect's frame orientation
    kinect_camera = dict(
        up=dict(x=-0.022634221477168148, y=0.9068314497962958, z=-0.42088527377240326),
        center=dict(
            x=0.21556439444543313, y=-0.21508937010308798, z=-0.08172521732195626
        ),
        eye=dict(x=0.12639005765507016, y=0.6196972437489059, z=1.7216858172237366),
    )
    # Add the basic plotly layout configuration
    fig.update_layout(
        title_text="3D Joints",
        title_font_family="Arial",
        title_font_color="#000000",
        title_font_size=18,
        width=1139,
        height=527,
    )
    # Set all the axes parameters, based on plotly for Python
    # documentation
    fig.update_scenes(
        aspectratio_x=1.1544060464581725,
        aspectratio_y=1.0451293568680509,
        aspectratio_z=0.8288412562301049,
        xaxis_range=[-1, 1],
        xaxis_showgrid=False,
        xaxis_showbackground=False,
        xaxis_showline=False,
        xaxis_zeroline=True,
        xaxis_zerolinecolor="#FF002A",
        xaxis_title_font_family="Arial",
        xaxis_title_font_color="#000000",
        xaxis_title_font_size=14,
        yaxis_range=[-1, 1],
        yaxis_showgrid=False,
        yaxis_showbackground=False,
        yaxis_showline=False,
        yaxis_zeroline=True,
        yaxis_zerolinecolor="#FF002A",
        yaxis_title_font_family="Arial",
        yaxis_title_font_color="#000000",
        yaxis_title_font_size=14,
        zaxis_range=[-1.5, -0.1],
        zaxis_showgrid=False,
        zaxis_showbackground=False,
        zaxis_showline=False,
        zaxis_zeroline=True,
        zaxis_zerolinecolor="#FF002A",
        zaxis_title_font_family="Arial",
        zaxis_title_font_color="#000000",
        zaxis_title_font_size=14,
        camera=kinect_camera,
    )
    # Since there are some troubles rendering with other
    # methods, we convert the plotly chart into a jpg image
    # in bytes
    img_bytes = fig.to_image(format="jpg")
    # We open the generated image with PIL
    img = Image.open(io.BytesIO(img_bytes))
    # We convert the image to an array, so open cv can
    # interpret it
    img_arr = np.array(img)
    cv2.imshow("joints", img_arr)
    cv2.waitKey(1)
