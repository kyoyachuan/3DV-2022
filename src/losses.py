import torch

from pytorch3d.loss import mesh_laplacian_smoothing


bce_loss = torch.nn.BCELoss()


# define losses
def voxel_loss(voxel_src, voxel_tgt):
	prob_loss = bce_loss(voxel_src, voxel_tgt.float())
	return prob_loss


def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	# implement laplacian smoothening loss
	return loss_laplacian


class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, w1=1.0, w2=1.0, each_batch=False):
        self.check_parameters(points1)
        self.check_parameters(points2)

        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist1 = torch.sqrt(dist1)**2
        dist2 = torch.sqrt(dist2)**2

        dist_min1, indices1 = torch.min(dist1, dim=2)
        dist_min2, indices2 = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean(1)
        loss2 = dist_min2.mean(1)
        
        loss = w1 * loss1 + w2 * loss2

        if not each_batch:
            loss = loss.mean()

        return loss

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(-1) == 3