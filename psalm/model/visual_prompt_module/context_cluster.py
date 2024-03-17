from abc import ABC, abstractmethod
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

def rand_sample(x, max_len):
    if x.shape[0] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
    return x[rand_idx, :]


def rand_sample_repeat(x, max_len):
    if x.shape[0] < max_len:
        indices = torch.randint(0, x.shape[0], (max_len - x.shape[0],))
        # pdb.set_trace()
        return torch.cat((x, x[indices]), dim=0)
    elif x.shape[0] == max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
        return x[rand_idx, :]


def point_sample(input, point_coords, return_dtype, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    output = F.grid_sample(input.float(), (2.0 * point_coords - 1.0).float().to(input.device), **kwargs)
    output = output.to(return_dtype)
    if add_dim:
        output = output.squeeze(3)
    return output


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class ConvReLULN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvReLULN1D, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            self.act
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # (B, C, N) -> (B, C_1, N)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        return x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class GeoRegionSampler(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_init_point,
                 num_sub_point,
                 num_neighbor,
                 pooler_mode='mean'):
        super(GeoRegionSampler, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_init_point = num_init_point
        self.num_sub_point = num_sub_point
        self.num_neighbor = num_neighbor

        self.diff_projector_list = nn.ModuleList()
        self.agg_projector_list = nn.ModuleList()
        self.pooler_list = nn.ModuleList()

        for ii in range(len(num_sub_point)):
            self.diff_projector_list.append(nn.Linear(self.input_dim + 2, self.input_dim + 2))
            self.agg_projector_list.append(ConvReLULN1D(in_channels=2 * (self.input_dim + 2),
                                                        out_channels=self.input_dim,
                                                        ))
            if pooler_mode == 'mean':
                self.pooler_list.append(nn.AvgPool1d(kernel_size=num_neighbor[ii]))
            elif pooler_mode == 'max':
                self.pooler_list.append(nn.AdaptiveMaxPool1d(output_size=1))
            else:
                raise NotImplementedError(f'{self.pooler_mode} is not supported.')

        self.flatten_projector = nn.Linear(self.input_dim * num_sub_point[-1], self.input_dim)
        self.dim_projector = nn.Linear(self.input_dim, self.output_dim)

        self.norm_init_weights()

    #  self.dtype = torch.float32
    def norm_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

    def forward(self,
                feature_map,
                region_masks,
                original_dtype,
                return_dtype):

        assert len(feature_map) == len(region_masks)

        all_points = []
        all_points_fea = []
        all_points_img_ids = []
        # Sample points and their features
        for img_idx, (region_feature_map_i, region_masks_list_i) in enumerate(zip(feature_map, region_masks)):
            if len(region_masks_list_i) != 0:
                # (w, h)
                ori_image_wh = torch.tensor([region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]],
                                            device=region_masks_list_i[0].device)[None,]
                # list of elements of shape [num_sample_point, 2]
                cur_non_zero_pos = [rand_sample_repeat((m.nonzero() / ori_image_wh), self.num_init_point) for m in
                                    region_masks_list_i]
                # list -> [num_mask, num_sample_point, 2]
                cur_non_zero_pos = torch.stack(cur_non_zero_pos)
                # [HxW, C] -> [H, W, C] -> [C, H, W] -> [N, C, H, W]
                h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                c = region_feature_map_i.shape[-1]
                dup_region_feature_map_i = region_feature_map_i.reshape(h, w, c).permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(cur_non_zero_pos.shape[0], 1, 1,
                                                                                        1)
                # [num_mask, C, H, W] x [num_mask, num_sample_point, 2] -> [num_mask, C, num_sample_point] -> [num_mask, num_sample_point, C]
                # F.grid_sample doesn't support BF16. Need to tranform into float32 then transform back.
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                region_feature_i = point_sample(dup_region_feature_map_i_ori_type,
                                                cur_non_zero_pos.flip(dims=(2,)).type(original_dtype),
                                                return_dtype,
                                                align_corners=True,
                                                )
                # region_feature_i = region_feature_i.to(dup_region_feature_map_i.dtype)
                region_feature_i = region_feature_i.transpose(-2, -1)

                cur_img_ids = [img_idx] * len(cur_non_zero_pos)
                # save to global list
                all_points.append(cur_non_zero_pos)
                all_points_fea.append(region_feature_i)
                all_points_img_ids.extend(cur_img_ids)

        # pdb.set_trace()
        # No region found, return list of None.
        if len(all_points) == 0:
            return [None] * len(region_masks)

        all_points = torch.cat(all_points, dim=0).to(return_dtype)  # [B*num_mask, num_sample_point, 2]
        all_points_fea = torch.cat(all_points_fea, dim=0)  # [B*num_mask, num_sample_point, C]
        all_points_img_ids = torch.tensor(all_points_img_ids, device=all_points_fea.device)
        # pdb.set_trace()
        assert all_points_fea.shape[:-1] == all_points_fea.shape[:-1]

        # Processing.
        for stage_i in range(len(self.num_sub_point)):
            cur_num_sub_point = self.num_sub_point[stage_i]
            cur_num_neighbor = self.num_neighbor[stage_i]

            all_points = all_points.contiguous()  # xy [btach, points, xy]
            fps_idx = farthest_point_sample(all_points, cur_num_sub_point).long()

            new_points = index_points(all_points, fps_idx)  # [B, npoint, 2]
            new_points_fea = index_points(all_points_fea, fps_idx)  # [B, npoint, d]

            idx = knn_point(cur_num_neighbor, all_points, new_points)
            grouped_points = index_points(all_points, idx)  # [B, npoint, k, 2]
            grouped_points_fea = index_points(all_points_fea, idx)  # [B, npoint, k, d]

            # pdb.set_trace()
            local_points_fea = torch.cat([grouped_points_fea, grouped_points], dim=-1)  # [B, npoint, k, d+2]
            anchor_points_fea = torch.cat([new_points_fea, new_points], dim=-1).unsqueeze(-2)
            diff_points_fea = local_points_fea - anchor_points_fea

            diff_points_fea = self.diff_projector_list[stage_i](diff_points_fea)
            gather_points_fea = torch.cat([diff_points_fea, anchor_points_fea.repeat(1, 1, cur_num_neighbor, 1)],
                                          dim=-1)  # [B, npoint, k, 2(d+2)]

            # pdb.set_trace()
            b, n, s, d = gather_points_fea.size()
            gather_points_fea = gather_points_fea.permute(0, 1, 3, 2)  # [B, npoint, 2(d+2), k]
            gather_points_fea = gather_points_fea.reshape(-1, d, s)  # [B*npoint, 2(d+2), k]
            gather_points_fea = self.agg_projector_list[stage_i](gather_points_fea)  # [B*npoint, d, k]
            # pdb.set_trace()
            batch_size, new_dim, _ = gather_points_fea.size()
            gather_points_fea = self.pooler_list[stage_i](gather_points_fea).view(batch_size, new_dim)  # [B*npoint, d]
            # gather_points_fea = F.adaptive_max_pool1d(gather_points_fea, 1).view(batch_size, -1) # [B*npoint, d]
            # pdb.set_trace()
            gather_points_fea = gather_points_fea.reshape(b, n, -1)  # [B, npoint, d]
            # pdb.set_trace()

            all_points = new_points
            all_points_fea = gather_points_fea

        # pdb.set_trace()
        x = all_points_fea.flatten(1, -1)  # [B, npoint x d]
        x = self.flatten_projector(x)
        all_region_fea = self.dim_projector(x)  # [B, d]

        output_region_fea = []
        for img_idx in range(len(region_masks)):
            cur_mask = all_points_img_ids == img_idx
            # pdb.set_trace()
            if not cur_mask.any():
                output_region_fea.append(None)
            else:
                output_region_fea.append(all_region_fea[cur_mask])

        # pdb.set_trace()
        return output_region_fea



class region_pooling(nn.Module):
    def __init__(self, num_sample_point):
        super().__init__()
        self.num_sample_point = num_sample_point
        self.pooler = nn.AdaptiveAvgPool1d(output_size=1)

    def extract_region_feature(self, region_feature_map, region_masks, original_dtype, return_dtype):
        assert len(region_feature_map) == len(region_masks)
        all_points = []
        all_points_fea = []
        all_points_img_ids = []
        for img_id, (region_feature_map_i, region_masks_list_i) in enumerate(zip(region_feature_map, region_masks)):
            # [H*W, C]
            if len(region_masks_list_i) != 0:
                ori_image_wh = torch.tensor([region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]], device=region_masks_list_i[0].device)[None,]
                # [num_sample_point, 2]
                for m in region_masks_list_i:
                    if m.nonzero().shape[0] <=0:
                        print('error')
                cur_non_zero_pos = [rand_sample_repeat((m.nonzero() / ori_image_wh), self.num_sample_point) for m
                                    in
                                    region_masks_list_i]
                # [num_mask, num_sample_point, 2]
                cur_non_zero_pos = torch.stack(cur_non_zero_pos)

                h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                c = region_feature_map_i.shape[-1]

                dup_region_feature_map_i = region_feature_map_i.reshape(h, w, c).permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(cur_non_zero_pos.shape[0], 1, 1,
                                                                                        1)
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                region_feature_i = point_sample(dup_region_feature_map_i_ori_type,
                                                cur_non_zero_pos.flip(dims=(2,)).type(original_dtype),
                                                return_dtype,
                                                align_corners=True,
                                                )
                # [num_mask, num_sample_point, C]
                region_feature_i = region_feature_i.transpose(-2, -1)

                cur_img_id = [img_id] * len(cur_non_zero_pos)

                all_points.append(cur_non_zero_pos)
                all_points_fea.append(region_feature_i)
                all_points_img_ids.extend(cur_img_id)

        return all_points, all_points_fea, all_points_img_ids

    def forward(self, feature_map, region_masks, original_dtype, return_dtype):
        assert len(feature_map) == len(region_masks)
        batch_size = len(feature_map)
        all_points, all_points_fea, all_points_img_ids = self.extract_region_feature(feature_map, region_masks,
                                                                                     original_dtype, return_dtype)

        if len(all_points) == 0:
            return [None] * len(region_masks)

        all_points = torch.cat(all_points, dim=0).to(return_dtype)
        all_points_fea = torch.cat(all_points_fea, dim=0).to(return_dtype)
        all_points_img_ids = torch.tensor(all_points_img_ids, device=all_points_fea.device)

        region_feat = self.pooler(all_points_fea.transpose(-2, -1)).transpose(-2, -1)

        region_feature_list = []
        for bs in range(batch_size):
            index = all_points_img_ids == bs
            region_feature_list.append(region_feat[index])
        return region_feature_list










