import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

from pytorch3d.ops import sample_points_from_meshes


mpl.rcParams['figure.dpi'] = 80


def plot(img, pred, gt, cfg, title=None):
    if cfg.dtype == 'point':
        plot_pointcloud(img, pred, gt, title)
    elif cfg.dtype == 'voxel':
        plot_voxel(img, pred, gt, title)
    elif cfg.dtype == 'mesh':
        pred_pc = sample_points_from_meshes(pred, cfg.n_points)
        gt_pc = sample_points_from_meshes(gt, cfg.n_points)
        plot_pointcloud(img, pred_pc, gt_pc, title)


def plot_pointcloud(img, pred, gt, title=None):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.3))
    if title:
        fig.suptitle(title, fontsize=14)

    # image plot
    ax = fig.add_subplot(1, 3, 1)
    img = img.detach().cpu().numpy()
    ax.imshow(img)
    ax.set_title('Image')

    # prediction plot
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    x, y, z = pred.clone().detach().cpu().squeeze().unbind(1) 
    ax.scatter3D(x, z, -y, s=5)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('Pred')
    ax.view_init(190, 190)

    # ground truth plot
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    xg, yg, zg = gt.clone().detach().cpu().squeeze().unbind(1) 
    ax.scatter3D(xg, zg, -yg, s=5)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('GT')
    ax.view_init(190, 190)

    plt.show()


def plot_voxel(img, pred, gt, title=None):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.3))
    if title:
        fig.suptitle(title, fontsize=14)

    # image plot
    ax = fig.add_subplot(1, 3, 1)
    img = img.detach().cpu().numpy()
    ax.imshow(img)
    ax.set_title('Image')

    # prediction plot
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    voxel = pred.clone().detach().cpu().permute(0,2,1)
    ax.voxels((voxel>0.5))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    ax.set_title('Pred')
    ax.view_init(190, 190)

    # ground truth plot
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    voxel_gt = gt.clone().detach().cpu().permute(0,2,1)
    ax.voxels(voxel_gt)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    ax.set_title('GT')
    ax.view_init(190, 190)

    plt.show()