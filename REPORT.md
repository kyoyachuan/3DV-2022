# Assignment: Single View to 3D

```
> name: 周俊川 
> id: 310551002
```

## Abstract

In this assignment, we are going to explore single view 3D prediction, include point cloud, mesh and voxel prediction. We are going to build different 3D representation modeling by leveraging **Pytorch3d** and its data structure, implement its forward function and visualize method, and try some hyperparameter for better result in point cloud prediction.


## Code Structure & Implementation

The overall code structure and implementations include

- configs
    - config.yml: Configuration setup for the train and eval execution, including
        - dtype: for specify 3D representation (point, mesh or voxel)
        - n_points: hyperparameter for point and mesh
        - dim_voxels: dimension of voxel which use in voxel dtype
        - w_chamfer: weight of chamfer loss
        - w_smooth: weight of laplacian smoothing loss
- src
    - dataset.py: Custom dataset definition for loading and transforming point clouds, meshes and voxels. **Custom meshes batch collection function needs to be implemented with `join_meshes_as_batch` as the default batch collect method does not support *Mesh* object**.
    - losses.py: Custom loss function, including chamfer distance, laplacian smoothing loss, voxel loss (which leverage binary cross entropy)
    - model.py: Custom model module alongs with each 3D representation decoder network. The decoder network is consists of several fully connected layer. The mesh prediction needs another source mesh (we use sphere) for deformation.
    - visualize.py: Custom visualization method for each of 3D representation. We use point cloud visualization method to visualize mesh result with `sample_points_from_meshes`. Note that we swap *y* and *z* axis for a better visualization view.
- eval.py: evaluation script
- train.py: training script

## Result & Analysis

### Dataset & Experiment

Our experiment use **chair** from **ShapeNet** dataset as our model learning target dataset. There is consisting 6778 samples.

This experiments was executed in google colab. We adopt 10 of batch size, 0.0004 of learning rate, 10000 iteration. We use **Resnet18** as our base encoder. 

### Point Cloud

We tried different `n_points` to see the performance behaviour. We experimented *1024*, *2048*, and *5000* output points. We found that chamfer distance will get lower as more points generated.

| n_points | chamfer distance |
| --------: | --------: |
| 1024     | 0.007     |
| 2048     | 0.006    |
| 5000     | 0.005    |


| image | ground truth | 1024 points | 2048 points | 5000 points |
| :--------: | :--------: | :--------: | :--------: | :--------: |
| ![](https://i.imgur.com/xKEbBsu.png) | ![](https://i.imgur.com/NOSIEJU.png) | ![](https://i.imgur.com/cO3RU7s.png) | ![](https://i.imgur.com/I54BgXU.png) | ![](https://i.imgur.com/5ZLq7wa.png) |
| ![](https://i.imgur.com/iB0rdu6.png) | ![](https://i.imgur.com/18Mgjjn.png) | ![](https://i.imgur.com/Fr8k4vt.png) | ![](https://i.imgur.com/iLu9IWF.png) | ![](https://i.imgur.com/gnX79LM.png) |
| ![](https://i.imgur.com/GLDn24N.png) | ![](https://i.imgur.com/Rm1C83Y.png) | ![](https://i.imgur.com/EqWwkty.png) | ![](https://i.imgur.com/t5KiRvk.png) | ![](https://i.imgur.com/lEHPOhp.png) |
| ![](https://i.imgur.com/iY5vCsK.png) | ![](https://i.imgur.com/U8QUbeN.png) | ![](https://i.imgur.com/DKGx8ky.png) | ![](https://i.imgur.com/61X4Xck.png) | ![](https://i.imgur.com/DW1weg2.png) |

### Voxel

The dataset's voxel dimension is 33x33x33. The training result shows a good result while the chair has more thick structure, but a bad result while the chair has more thin structure.

Some good cases with the chair has more thick structure
![](https://i.imgur.com/msHeoeM.png)
![](https://i.imgur.com/4n5RV5K.png)
![](https://i.imgur.com/XcRV95A.png)

Some failure cases with the chair has more thin structure
![](https://i.imgur.com/gjrSzSs.png)
![](https://i.imgur.com/srWeU4P.png)
![](https://i.imgur.com/UTTVDSz.png)

### Mesh

We use sphere as initial source mesh and learn a deform matrix to deform the sphere to target mesh. We train mesh prediction with sample 2048 points and with weight 0.1 laplacian smoothing regularization.

The fundamental result shows the model has trained to deform a sphere to chair-like shape, but it is poor at all compare with ground truth. 

![](https://i.imgur.com/N306xQe.png)
![](https://i.imgur.com/wXyWDeu.png)
![](https://i.imgur.com/aS1pekS.png)


## Conclusion
In this assignment, we explore 3 kind of 3D representation prediction model: point clouds, meshes, and voxels. We implemented each of its corresponding dataloader, loss function, decoder and forward function, visualization method.

We tried different `n_points` for point cloud prediction modeling and found that more `n_points` can achieve better result. For the voxel prediction model, we have a good result in thick structure of chair but a poor result in thin structure of chair. For the mesh prediction model, we trained the model which able to deform a sphere to chair-like shape but is still poor at all compare with ground truth.


## Reference
- https://github.com/nctu-eva-lab/3DV-2022
- https://pytorch3d.readthedocs.io/en/latest/

