import numpy as np
import random
import os
import copy
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
sns.reset_orig()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def transform_points(points, transformation):
    points_homo = np.concatenate([points[:, 0:3], np.ones((len(points), 1))], axis=1)
    icp_points = transformation @ points_homo.T  # [4, n]
    return icp_points.T[:, 0:3] 

def visualize_pcd(points, labels=None, num_colors=3, title='visualization', if_save=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    if labels is None:
        pass
    else:
        COLOR_MAP = sns.color_palette('husl', n_colors=num_colors)
        COLOR_MAP = np.array(COLOR_MAP)
        # print('COLOR_MAP: ', COLOR_MAP)
        labels = labels.astype(int)
        colors = COLOR_MAP[labels%len(COLOR_MAP)]
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if if_save:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
        vis.get_render_option().point_size = 3.0
        vis.update_renderer()
        vis.capture_screen_image(filename=os.path.join(f'ego_motion_o3d.png'), do_render=True)
        vis.destroy_window()

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)
        # vis.add_geometry(pcd)
        # vis.update_geometry(pcd)
        # vis.poll_events()
        # vis.update_renderer()

        # image = vis.capture_screen_float_buffer(False)
        # image=np.asarray(image)*255
        # image=cv2.resize(image,(264,264))
        # image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # print(image.shape) 
        # cv2.imwrite("test.png",image)
        # vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_pcd_multiple(points1, points2, labels1, labels2, num_colors=100, title='', if_save=False):
    pcd = o3d.geometry.PointCloud()
    points3 = points2.copy()
    points3[:, 1] -= max(np.abs(points1[:, 1])) * 1.0  # move the bottom
    pcd.points = o3d.utility.Vector3dVector(np.concatenate((points1[:,:3], points3[:,:3]), axis=0))
    labels = np.concatenate((labels1.astype(int), labels2.astype(int)), axis=0)
    COLOR_MAP = sns.color_palette('husl', n_colors=num_colors)
    COLOR_MAP = np.array(COLOR_MAP)
    colors = COLOR_MAP[labels%len(COLOR_MAP)]
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    if if_save:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
        vis.get_render_option().point_size = 3.0
        vis.update_renderer()
        vis.capture_screen_image(filename=os.path.join(f'vis_two_pcds_o3d.png'), do_render=True)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd], window_name=title)

def visualize_pcd_plotly(lidar, label, num_colors=100, title=None):  
    '''
    #input: 
    lidar(Nx3)
    label(Nx1)
    '''
    COLOR_MAP = sns.color_palette('husl', num_colors)
    COLOR_MAP = np.array(COLOR_MAP)
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        # scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.2))
        scene=dict(aspectmode='data'), 
    )
    fig = go.Figure(layout=layout, layout_title_text="A Figure Displayed with the 'svg' Renderer")
    fig.add_trace(go.Scatter3d(
            x=lidar[:, 0],
            y=lidar[:, 1],
            z=lidar[:, 2],
            mode='markers',
            marker={
                'size': 1,
                'opacity': 0.8,
                'color': COLOR_MAP[label.astype(int)%num_colors].tolist(),
            }
        )
    )
    ann = []
    for l in np.unique(label.astype(int)):
        inds = label==l
        xyz = lidar[inds]
        xyz = xyz[np.random.randint(low=0, high=len(xyz), size=(1)).item()]
        ann.append(dict(x=xyz[0], y=xyz[1], z=xyz[2], text=str(l)))
    fig.update_layout(
        scene=dict(
            # xaxis=dict(type="date"),
            # yaxis=dict(type="category"),
            # zaxis=dict(type="log"),
            annotations=ann
        )
    )
    # for xyz, l in zip(lidar, label):
    #     fig.add_annotation(
    #             x=xyz[0],
    #             y=xyz[1],
    #             z=xyz[2],
    #             text=str(l),
    #             )
    fig.update_layout(
        title=dict(text='dummy title')
    )
    fig.show()