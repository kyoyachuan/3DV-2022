import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.dpi'] = 80


def plot(img, pred, gt, cfg, title=None):
    if cfg.dtype == 'point':
        plot_pointcloud(img, pred, gt, title)
    elif cfg.dtype == 'voxel':
        plot_voxel(img, pred, gt, title)
    elif cfg.dtype == 'mesh':
        plot_mesh(img, pred, gt, title)


def plot_mesh(img, pred, gt, title=None):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.3))
    if title:
        fig.suptitle(title, fontsize=14)

    ax = fig.add_subplot(1, 3, 1)
    img = img.detach().cpu().numpy()
    ax.imshow(img)
    ax.set_title('Image')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    x, y, z = pred.verts_packed().clone().detach().cpu().squeeze().unbind(1)
    tri = pred.faces_packed_to_mesh_idx().clone().detach().cpu()
    ax.plot_trisurf(x, z, -y, triangles=tri)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('Pred')
    ax.view_init(190, 190)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    xg, yg, zg = gt.verts_packed().clone().detach().cpu().squeeze().unbind(1) 
    trig = pred.faces_packed_to_mesh_idx().clone().detach().cpu()
    ax.plot_trisurf(xg, zg, -yg, triangles=trig)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('GT')
    ax.view_init(190, 190)

    plt.show()


def plot_pointcloud(img, pred, gt, title=None):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.3))
    if title:
        fig.suptitle(title, fontsize=14)

    ax = fig.add_subplot(1, 3, 1)
    img = img.detach().cpu().numpy()
    ax.imshow(img)
    ax.set_title('Image')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    x, y, z = pred.clone().detach().cpu().squeeze().unbind(1) 
    ax.scatter3D(x, z, -y, s=5)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('Pred')
    ax.view_init(190, 190)

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

    ax = fig.add_subplot(1, 3, 1)
    img = img.detach().cpu().numpy()
    ax.imshow(img)
    ax.set_title('Image')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    voxel = pred.clone().detach().cpu().permute(0,2,1)
    ax.voxels((voxel>0.5))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    ax.set_title('Pred')
    ax.view_init(190, 190)

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