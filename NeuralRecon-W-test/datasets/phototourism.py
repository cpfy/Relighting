import open3d as o3d
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import cv2
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from .ray_utils import *
from .colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

import yaml
from tools.prepare_data.generate_voxel import get_near_far, gen_octree_from_sfm
import time
from kornia import create_meshgrid
import h5py

# additional configuration
sfm_path = "sparse"
vis_octree = False
vis_intersection = False
vis_depth = False
depth_percent = 0
skip = 1


# 旅游照片集：关于数据读取、处理
# 好家伙，原来这个是从nerf_pl的nerfw分支下面同名文件夹+文件改的
class PhototourismDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split="train",
            img_downscale=1,
            val_num=1,
            use_cache=False,
            cache_paths=["cache"],
            split_path="",
            semantic_map_path=None,
            with_semantics=True,
            use_voxel=True,
            scene_origin=None,
            scene_radius=None,
            shared_cache=False,
            shared_rays_base=None,
            shared_rgbs_base=None,
            all_rays_shape=None,
            all_rgbs_shape=None,
    ):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        semantic_map_path: path to the directory which stores the semantic map
        with_semantics: if give semantic labels when get item
        """
        self.split_path = split_path
        self.root_dir = root_dir
        self.split = split  # [指用途区分]某一个数据被分到train、eval、test_train、val、test五个中哪一类

        # 图像W、H降低规模的倍率，原始d=1在BG上将消耗40G内存？！
        assert (
                img_downscale >= 1
        ), "image can only be downsampled, please set img_downscale>=1!"
        self.img_downscale = img_downscale
        if split == "val":  # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(8, self.img_downscale)
        self.val_num = max(1, val_num)  # at least 1
        self.define_transforms()
        self.white_back = False
        self.semantic_map_path = semantic_map_path  # 语义地图映射的路径，后文读取npz用
        self.with_semantics = with_semantics

        self.scene_origin = scene_origin
        self.scene_radius = scene_radius

        # hard code sfm depth padding
        scene_name = self.root_dir.rsplit('/')[-1]
        if scene_name == 'brandenburg_gate':
            sfm_path = '../neuralsfm'
            depth_percent = 0.2
        elif scene_name == 'palacio_de_bellas_artes':
            sfm_path = '../neuralsfm'
            depth_percent = 0.4
        elif scene_name in ['lincoln_memorial', 'pantheon_exterior']:
            sfm_path = '../neuralsfm'  # 原repo未加此行
            depth_percent = 0.0
        else:
            sfm_path = '../neuralsfm'
            depth_percent = 0.0  # <修改>防止depth_percent未赋值就使用。如非训练or非四场景or仅评估ckpt情况会使用

        # sfm深度填充比例？和不同数据集特性有关
        # 与nerf_pl比较文学可知，仅train、eval状态且不使用cache会用到此参数
        self.depth_percent = depth_percent

        # 3D点云
        self.sfm_path = sfm_path

        print(f"reading sfm result from {self.sfm_path}...")  # [输出]尝试读取sfm数据

        # Setup cache
        self.use_cache = use_cache
        if self.use_cache:
            assert self.split == "train", "only can use cache during training"
        # sparse_voxel
        self.octree_data = None
        self.use_voxel = use_voxel
        self.use_voxel = False  # todo [修改]可能这里导致了额外一步mask，使all_rays维度出现问题
        if self.use_voxel and not self.use_cache and self.split == "train":
            print("Note: training near far will generate from sparse voxel!!!!")  # [输出] 显然train时出现了这一问题

        self.cache_paths = cache_paths
        self.shared_cache = shared_cache

        # 默认为False，不用管，可能是multi gpus用到
        if shared_cache:
            assert (
                    shared_rgbs_base is not None
            ), f"Empty shared array at rank {torch.distributed.get_rank()}"
            shared_array_rgb = np.ctypeslib.as_array(
                shared_rgbs_base.get_obj()
            ).reshape(all_rgbs_shape)
            self.all_rgbs = torch.from_numpy(shared_array_rgb)
            print("Cached rgbs loaded to torch!")
            shared_array_ray = np.ctypeslib.as_array(
                shared_rays_base.get_obj()
            ).reshape(all_rays_shape)
            self.all_rays = torch.from_numpy(shared_array_ray)
            print("Cached rays loaded to torch!")
        self.read_meta()

    # 似乎是观察视点所在球体，此函数后废弃
    def vis_sphere(self, center_id=47129, radius=4.6):
        pts3d = read_points3d_binary(
            os.path.join(self.root_dir, f"dense/{self.sfm_path}/points3D.bin")
        )
        xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
        print("pts3d at 47129", xyz_world[center_id + 1], xyz_world.shape)

        # center = xyz_world[center_id]
        center = np.array([0.568699, -0.0935532, 6.28958])
        dist = np.linalg.norm(xyz_world - center, axis=-1)
        filtered = xyz_world[dist < radius]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_world)
        o3d.io.write_point_cloud(f"origin_{self.root_dir.split('/')[-1]}.ply", pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered)
        o3d.io.write_point_cloud(
            f"unit_sphere_{self.root_dir.split('/')[-1]}_{center}_{radius}.ply", pcd
        )
        # max_far: 49.02846145629883, scale_factor: 9.805692291259765

        exit(0)

    # 返回每张图片的深度、权重（衡量图像数据有效性/质量）、归一化的方向向量
    def get_colmap_depth(
            self,
            img_p3d_all,
            img_2d_all,
            img_err_all,
            pose,
            intrinsic,
            img_w,
            img_h,
            device=0,
    ):
        # return depth and weights for each image
        # calculate normalize factor
        grid = create_meshgrid(
            img_h, img_w, normalized_coordinates=False, device=device
        )[0]
        i, j = grid.unbind(-1)
        fx, fy, cx, cy = (
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        directions = torch.stack(
            [(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1
        )  # (H, W, 3)

        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ pose[:, :3].T  # (H, W, 3)
        dir_norm = torch.norm(rays_d, dim=-1, keepdim=True).reshape(img_h, img_w)

        # depth from sfm key points
        depth_all = torch.zeros(img_h, img_w, device=device)
        weights_all = torch.zeros(img_h, img_w, device=device)

        img_2d_all = torch.round(img_2d_all).long()  # (width, height)
        valid_mask = (
                (img_2d_all[:, 0] >= 0)
                & (img_2d_all[:, 0] < img_w)
                & (img_2d_all[:, 1] >= 0)
                & (img_2d_all[:, 1] < img_h)
        )

        img_2d = img_2d_all[valid_mask]
        img_err = img_err_all[valid_mask].squeeze()

        img_p3d = img_p3d_all[valid_mask]
        pose = torch.cat((pose, torch.zeros(1, 4, device=device)), dim=0)
        pose[3, 3] = 1
        extrinsic = torch.linalg.inv(pose)

        Err_mean = torch.mean(img_err)
        projected = intrinsic @ extrinsic[:3] @ img_p3d.T

        depth = projected[2, :]
        weight = 2 * torch.exp(-((img_err / Err_mean) ** 2))  # 错误比率大的图片分配权重指数级小

        depth_all[img_2d[:, 1], img_2d[:, 0]] = depth
        weights_all[img_2d[:, 1], img_2d[:, 0]] = weight
        return depth_all.cpu() * dir_norm.cpu(), weights_all.cpu()

    # 从sfm点云建立八叉树，建模时voxel扩展可提升精度
    def get_octree(self, device, expand, radius=1):
        scene_config_path = os.path.join(self.root_dir, "config.yaml")
        with open(scene_config_path, "r") as yamlfile:
            scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        # decompose voxel config
        min_track_length = scene_config["min_track_length"]
        voxel_size = scene_config["voxel_size"]

        octree, scene_origin, scale, level = gen_octree_from_sfm(
            self.root_dir,
            min_track_length,
            voxel_size,
            device=device,
            visualize=vis_octree,
            expand=expand,
            radius=radius,
        )

        octree_data = {}
        octree_data["octree"] = octree
        octree_data["scene_origin"] = torch.from_numpy(scene_origin).cuda()
        octree_data["scale"] = scale
        octree_data["level"] = level
        return octree_data

    # 计算光线的最近、最远相交voxel
    # 调用generate_voxel中的get_near_far函数
    def near_far_voxel(self, octree_data, rays_o, rays_d, image_name, chunk_size=65536):
        """generate near for from intersection with sparse voxel.
           Input and output are in sfm coordinate system

        Args:
            rays_o (float tensor): ray origin
            rays_d (float tensor): ray direction
            chunk_size (int, optional): ray chunk size to intersect. Defaults to 1024.

        Returns:
            float tensor: near of all rays
            float tensor: far of all rays
            bool tensor: if the ray intersects with some voxel
        """

        # read config file
        scene_config_path = os.path.join(self.root_dir, "config.yaml")
        with open(scene_config_path, "r") as yamlfile:
            scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        gt_min = np.array(scene_config["eval_bbx"][0])
        gt_max = np.array(scene_config["eval_bbx"][1])
        scene_center = gt_min + (gt_max - gt_min) / 2

        # decompose voxel config
        min_track_length = scene_config["min_track_length"]
        voxel_size = scene_config["voxel_size"]

        # cuda device
        device = 0
        rays_o = rays_o.to(device).float()
        rays_d = rays_d.to(device).float()

        # unpack octree
        octree = octree_data["octree"]
        octree_origin = octree_data["scene_origin"].float()
        octree_scale = octree_data["scale"]
        octree_level = octree_data["level"]

        # 推测是这里两个list导致RAM爆炸
        voxel_near_sfm_all = []
        voxel_far_sfm_all = []

        # todo: figure out why chunck size greater or equal than 1768500 will result in error intersection
        # use 1000000 as a threshold just to be safe
        # [测试]原为10w，看下5w会发生什么，RAM占用率如何变化。（无变化，这里应该只是分块大小不同，总量没变）
        chunk_size = min(rays_o.size()[0], 100000)
        try:

            # print(f"chunk_size={chunk_size}")               # 固定=100000
            # print(f"rays_o.size()[0]={rays_o.size()[0]}")   # 震惊，是60w-80w中的一个随机数

            for i in tqdm(range(0, rays_o.size()[0], chunk_size)):
                # generate near far from spc
                voxel_near_sfm, voxel_far_sfm = get_near_far(
                    rays_o[i: i + chunk_size],
                    rays_d[i: i + chunk_size],
                    octree,
                    octree_origin,
                    octree_scale,
                    octree_level,
                    visualize=vis_intersection,
                    ind=f"cache_{image_name}_{i}",
                )
                voxel_near_sfm_all.append(voxel_near_sfm.cpu())
                voxel_far_sfm_all.append(voxel_far_sfm.cpu())
        except Exception as e:
            e.args = (
                    "This probably due to chunk_size too big, consider lower value! Original message:\n"
                    + e.args
            )
            raise

        voxel_near_sfm_all = torch.cat(voxel_near_sfm_all, dim=0).reshape(-1, 1)
        voxel_far_sfm_all = torch.cat(voxel_far_sfm_all, dim=0).reshape(-1, 1)

        valid_mask = voxel_near_sfm_all > 0

        # add far with voxel size
        voxel_far_sfm_all[valid_mask] = voxel_far_sfm_all[valid_mask] + voxel_size

        return (
            voxel_near_sfm_all.squeeze(),
            voxel_far_sfm_all.squeeze(),
            valid_mask.squeeze(),
        )

    # tsv读取 + train/test划分
    # use_cache的情况下全部不用处理，cache里已经搞好了
    def read_meta(self):
        # self.vis_sphere()
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, "*.tsv"))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep="\t")
        # we accept all images
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            # with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
            #     self.img_ids = pickle.load(f)
            # with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
            #     self.image_paths = pickle.load(f)
            pass
        else:
            # 读取dense/sparse/images.bin的稀疏BG图像数据？
            print("Reading images.bin..")
            imdata = read_images_binary(
                os.path.join(self.root_dir, f"dense/{self.sfm_path}/images.bin")
            )
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {}  # {id: filename}
            for filename in list(self.files["filename"]):
                if filename not in img_path_to_id.keys():
                    print(f"image {filename} not found in sfm result!!")
                    continue
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            # with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
            #     self.Ks = pickle.load(f)
            pass
        else:
            # 读取dense/sparse/cameras.bin的相机焦距、偏移等数据
            self.Ks = {}  # {id: K}
            print("Reading cameras.bin..")
            camdata = read_cameras_binary(
                os.path.join(self.root_dir, f"dense/{self.sfm_path}/cameras.bin")
            )
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[imdata[id_].camera_id]

                # 针孔相机
                if cam.model == "PINHOLE":
                    img_w, img_h = int(cam.params[2] * 2), int(cam.params[3] * 2)
                    img_w_, img_h_ = (
                        img_w // self.img_downscale,
                        img_h // self.img_downscale,
                    )
                    K[0, 0] = cam.params[0] * img_w_ / img_w  # fx
                    K[1, 1] = cam.params[1] * img_h_ / img_h  # fy
                    K[0, 2] = cam.params[2] * img_w_ / img_w  # cx
                    K[1, 2] = cam.params[3] * img_h_ / img_h  # cy
                    K[2, 2] = 1

                # 放射相机？
                elif cam.model == "SIMPLE_RADIAL":
                    img_w, img_h = int(cam.params[1] * 2), int(cam.params[2] * 2)
                    img_w_, img_h_ = (
                        img_w // self.img_downscale,
                        img_h // self.img_downscale,
                    )
                    K[0, 0] = cam.params[0] * img_w_ / img_w  # f
                    K[1, 1] = cam.params[0] * img_h_ / img_h  # f
                    K[0, 2] = cam.params[1] * img_w_ / img_w  # cx
                    K[1, 2] = cam.params[2] * img_h_ / img_h  # cy
                    K[2, 2] = 1
                else:
                    raise NotImplementedError(f"Not supported camera model {cam.model}")
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            # self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
            pass
        else:
            # 计算camera-to-world的映射
            print("Compute c2w poses..")
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.0]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            # self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            # with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
            #     self.nears = pickle.load(f)
            # with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
            #     self.fars = pickle.load(f)
            pass
        else:
            # 读取dense/sparse/points3D.bin的3D点云数据
            pts3d = read_points3d_binary(
                os.path.join(self.root_dir, f"dense/{self.sfm_path}/points3D.bin")
            )
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate(
                [self.xyz_world, np.ones((len(self.xyz_world), 1))], -1
            )
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {}  # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[
                            :, :3
                            ]  # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[
                    xyz_cam_i[:, 2] > 0
                    ]  # filter out points that lie behind the cam
                if self.scene_origin is not None:
                    scene_origin_h = np.concatenate(
                        [self.scene_origin, np.ones(1)], -1
                    )[np.newaxis, :]
                    origin_cam_i = (scene_origin_h @ w2c_mats[i].T)[:, :3]
                    self.nears[id_] = origin_cam_i[0, 2] - self.scene_radius * 1.5
                    self.fars[id_] = origin_cam_i[0, 2] + self.scene_radius * 1.5
                else:
                    self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                    self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()

        if not self.use_cache:
            self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

            # Step 5. split the img_ids (the number of images is verified to match that in the paper)
            # training will use all images
            self.img_ids_train = [
                                     id_
                                     for i, id_ in enumerate(self.img_ids)
                                     if not self.files.loc[i, "split"] == "test"
                                 ][::skip]
            self.img_ids_test = [
                id_
                for i, id_ in enumerate(self.img_ids)
                if self.files.loc[i, "split"] == "test"
            ]
            # self.img_ids_test = [1159]
            self.N_images_train = len(self.img_ids_train)
            self.N_images_test = len(self.img_ids_test)

        if self.split in ["train", "eval"]:  # create buffer of all rays and rgb data
            if self.use_cache:
                if self.shared_cache:
                    # Do nothing, initialized outside.
                    pass
                else:
                    # loading with splits
                    self.all_rays = []
                    self.all_rgbs = []
                    # determine cache type
                    example_split_path = os.path.join(
                        self.root_dir, self.split_path, self.cache_paths[0]
                    )
                    cache_type = os.listdir(example_split_path)[0].split(".")[-1]

                    # 从cache加载数据
                    for cache_path in self.cache_paths:
                        print(f"Loading cached rays from {cache_path}..")
                        rays_file = os.path.join(
                            self.root_dir,
                            self.split_path,
                            f"{cache_path}/rays{self.img_downscale}.{cache_type}",
                        )
                        if cache_type == "npz":
                            all_rays = np.load(rays_file)
                            self.all_rays += [torch.from_numpy(all_rays["arr_0"])]
                        elif cache_type == "h5":
                            f = h5py.File(rays_file, "r")
                            all_rays = f["rays"][:]
                            f.close()
                            self.all_rays += [torch.from_numpy(all_rays)]
                        print("Cached rays loaded!")

                        print(f"Loading cached rgbs from {cache_path}..")
                        rgbs_file = os.path.join(
                            self.root_dir,
                            self.split_path,
                            f"{cache_path}/rgbs{self.img_downscale}.{cache_type}",
                        )

                        if cache_type == "npz":
                            all_rgbs = np.load(rgbs_file)
                            self.all_rgbs += [torch.from_numpy(all_rgbs["arr_0"])]
                        elif cache_type == "h5":
                            f = h5py.File(rgbs_file, "r")
                            all_rgbs = f["rgbs"][:]
                            f.close()
                            self.all_rgbs += [torch.from_numpy(all_rgbs)]
                        print("Cached rgbs loaded!")
                    self.all_rays = torch.cat(self.all_rays, 0)
                    self.all_rgbs = torch.cat(self.all_rgbs, 0)
            else:
                # 生成光线与rgbs并放在缓存
                print("Generating rays and rgbs..")
                self.all_rays = []
                self.all_rgbs = []
                if self.split == "eval":
                    self.img_wh = []
                    self.eval_images = []
                    self.extrinsics = []
                    self.intrinsics = []
                img_ids_split = (
                    self.img_ids_train if self.split == "train" else self.img_ids_test
                )

                # key point depth
                pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4)
                error_array = torch.ones(max(pts3d.keys()) + 1, 1)
                for pts_id, pts in tqdm(pts3d.items()):
                    pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz)
                    error_array[pts_id, 0] = torch.from_numpy(pts.error)
                print("Mean Projection Error:", torch.mean(error_array))
                self.sfm_octree = self.get_octree(device=0, expand=1, radius=1)
                self.expand_octree = self.get_octree(device=0, expand=2, radius=1.5)

                # 此处打开1300+个照片，所谓use_cache。RAM也是被这个b搞爆的
                for id_ in tqdm(img_ids_split):
                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(
                        os.path.join(
                            self.root_dir, "dense/images", self.image_paths[id_]
                        )
                    ).convert("RGB")
                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w // self.img_downscale
                        img_h = img_h // self.img_downscale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    if self.split == "eval":
                        self.img_wh += [torch.LongTensor([img_w, img_h])]
                        self.eval_images += [self.image_paths[id_]]
                        self.extrinsics += [c2w]
                        self.intrinsics += [self.Ks[id_]]
                    img = self.transform(img)  # (3, h, w)
                    img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

                    directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)

                    # key point depth
                    # fix pose
                    img_colmap = imdata[id_]
                    pose = torch.FloatTensor(self.poses_dict[id_]).cuda()
                    pose[..., 1:3] *= -1
                    intrinsic = torch.FloatTensor(self.Ks[id_]).cuda()

                    valid_3d_mask = img_colmap.point3D_ids != -1
                    point3d_ids = img_colmap.point3D_ids[valid_3d_mask]
                    img_p3d = pts3d_array[point3d_ids].cuda()
                    img_err = error_array[point3d_ids].cuda()
                    img_2d = torch.from_numpy(img_colmap.xys)[valid_3d_mask]
                    depth_sfm, weight = self.get_colmap_depth(
                        img_p3d, img_2d, img_err, pose, intrinsic, img_w, img_h
                    )
                    depths = depth_sfm.reshape(-1, 1)
                    weights = weight.reshape(-1, 1)

                    image_name = self.image_paths[id_].split(".")[0]

                    if vis_depth:
                        print(f"saving... at samples/depth/{image_name}.ply")
                        os.makedirs("samples/depth", exist_ok=True)
                        pts = rays_o + rays_d * depths
                        gt_pcd = o3d.geometry.PointCloud()
                        gt_pcd.points = o3d.utility.Vector3dVector(pts.numpy())
                        o3d.io.write_point_cloud(
                            f"samples/depth/{image_name}.ply", gt_pcd
                        )

                    # BG的数据集照片基本都自带semantics
                    if self.with_semantics:
                        semantic_map = np.load(
                            os.path.join(
                                self.root_dir,
                                f"{self.semantic_map_path}/{image_name}.npz",
                            )
                        )["arr_0"]
                        semantic_map = cv2.resize(
                            semantic_map,
                            (
                                semantic_map.shape[1] // self.img_downscale,
                                semantic_map.shape[0] // self.img_downscale,
                            ),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        rays_mask = semantic_map.reshape(-1, 1)

                        rays = torch.cat(
                            [
                                rays_o,
                                rays_d,
                                self.nears[id_] * torch.ones_like(rays_o[:, :1]),
                                self.fars[id_] * torch.ones_like(rays_o[:, :1]),
                                rays_t,
                                torch.from_numpy(rays_mask),
                                depths,
                                weights,
                            ],
                            1,  # 指拼接的维数dim=1
                        )  # (h*w, 13)
                        # print(f"【Out】In line {656}，rays size={rays.size()}")  # [输出]每个image输出一次，格式均为[x,12]
                    else:
                        rays = torch.cat(
                            [
                                rays_o,
                                rays_d,
                                self.nears[id_] * torch.ones_like(rays_o[:, :1]),
                                self.fars[id_] * torch.ones_like(rays_o[:, :1]),
                                rays_t,
                                depths,
                                weights,
                            ],
                            1,  # 指拼接的维数dim=1
                        )  # (h*w, 12)
                        # print(f"【Out】In line {670}，rays size={rays.size()}")  # 不会输出，均为656

                    if self.split == "train" and self.use_voxel:
                        # replace near far with voxel intersection, and get rid of rays that don't intersect with any voxel
                        _, _, valid_mask = self.near_far_voxel(
                            self.sfm_octree, rays_o, rays_d, image_name
                        )
                        (
                            voxel_nears,
                            voxel_fars,
                            expand_valid_mask,
                        ) = self.near_far_voxel(
                            self.expand_octree, rays_o, rays_d, image_name
                        )

                        # assert torch.sum(~expand_valid_mask[valid_mask]) < 10, f"intersection invalid! expanded octree should cover smaller octree, num invalid: {torch.sum(~expand_valid_mask[valid_mask])}"

                        rays[valid_mask, 6] = voxel_nears[valid_mask]
                        rays[valid_mask, 7] = voxel_fars[valid_mask]

                        img = img[valid_mask]
                        rays = rays[valid_mask]

                    # 此处的函数输出在 data_generation.sh 运行时多次出现
                    # rays是[ray_o, ray_d, ..., depth, weight]等一大串拼起来
                    if self.depth_percent > 0:
                        valid_depth = rays[:, -2] > 0
                        if not valid_depth.any():  # [Solve]官方的除0报错解决方案
                            continue

                        valid_num = torch.sum(valid_depth).long().item()  # torch.Tensor.item(): 返回单个元素张量的值
                        current_len = rays.size()[0]

                        # 原写法除0报错: current_percent = valid_num / current_len
                        if current_len == 0:
                            current_percent = 2.22222
                        else:
                            current_percent = valid_num / current_len  # [报错]运行时有一个ZeroDivision报错

                        # 例如，对BG：pad_len = (0.2*len-?)/0.8
                        # 一般是个5w-15w之间的数
                        padding_length = int(
                            np.ceil((self.depth_percent * current_len - valid_num) / (1 - self.depth_percent)))
                        print(
                            f"padding valid depth percentage: from {current_percent} to {self.depth_percent} with padding {padding_length}")

                        pad_ind = torch.floor((torch.rand(padding_length) * valid_num)).long()
                        result_length = padding_length + current_len
                        result_ind = torch.randperm(result_length)

                        # [报错] 增大downscale会出现index越界问题
                        # 官方解释：https://github.com/zju3dv/NeuralRecon-W/issues/13
                        # The problem seems to be there is no SfM key point exists in that image. The latest code has fixed this issue.
                        paddings_rays = rays[valid_depth, :][pad_ind]
                        rays = torch.cat([rays, paddings_rays], dim=0)[result_ind]

                        paddings_rgbs = img[valid_depth, :][pad_ind]
                        img = torch.cat([img, paddings_rgbs], dim=0)[result_ind]

                        test_ind = torch.floor((torch.rand(1024) * result_length)).long()

                        # [报错]原写法除0报错: print(f"sample depth percent after padding: {torch.sum(rays[test_ind, -2] > 0) / rays[test_ind].size()[0]}")
                        if current_len == 0 or rays[test_ind].size()[0] == 0:
                            print("Find One Zero Division.")
                        else:
                            print(
                                f"sample depth percent after padding: {torch.sum(rays[test_ind, -2] > 0) / rays[test_ind].size()[0]}")

                    self.all_rgbs += [img]
                    self.all_rays += [rays]
                    print(f"【Size：rays】{rays.size()}")  # [输出]size = [8538,12]
                    print(f"【Len：all_rays】{len(self.all_rays)}")  # [报错]这是一个list，无法使用.size()方法

                # 如此重要的修正Tensor shape一行，哪个给注释了？？？导致train的shape一直报错
                if self.split == 'train':
                    self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 10)
                    self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)

        elif self.split in [
            "val",
            "test_train",
        ]:  # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]

        else:  # for testing, create a parametric rendering path
            # test poses and appearance index are defined in test.py
            pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split in ["train", "eval"]:
            return len(self.all_rays)
        if self.split == "test_train":
            return self.N_images_train
        if self.split == "val":
            return self.val_num
        return len(self.poses_test)  # 此时为test类别

    # 类似dataset[5]这种取值时的对应操作
    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers

            # 加上上面line 744的Tensor shape修改总算格式ok了
            # print(f"【Out】idx={idx}")    # [输出]121，100-200w等
            # print(f"【Out】Type of all_rays = ", end="")  # [输出]<class 'list'>    加上上面一行总算正常了
            # print(type(self.all_rays))
            # print(f"【Out】Len of all_rays = {len(self.all_rays)}")   # [输出]184/263等，修复正确后恒为2333642

            sample = {
                "rays": self.all_rays[idx, :8],  # [报错]list indices must be integers or slices, not tuple
                "ts": self.all_rays[idx, 8].long(),
                "rgbs": self.all_rgbs[idx],
            }
            if self.with_semantics:
                sample["semantics"] = self.all_rays[idx, 9]
                sample["rays"] = torch.cat(
                    (self.all_rays[idx, :8], self.all_rays[idx, 10:13]), dim=-1
                )
            else:
                sample["rays"] = torch.cat(
                    (self.all_rays[idx, :8], self.all_rays[idx, 9:12]), dim=-1
                )

        # 这是相比nerf_pl新加的，大概率evaluate sdf用
        elif self.split == "eval":
            w, h = self.img_wh[idx]
            all_rays = self.all_rays[idx].reshape(h, w, -1)
            all_rgbs = self.all_rgbs[idx].reshape(h, w, -1)
            left_rays = all_rays[:, : w // 2].reshape(-1, 9)
            right_rays = all_rays[:, w // 2:].reshape(-1, 9)
            left_rgbs = all_rgbs[:, : w // 2].reshape(-1, 3)
            right_rgbs = all_rgbs[:, w // 2:].reshape(-1, 3)
            sample = {
                "rays": self.all_rays[idx][:, :8],
                "ts": self.all_rays[idx][:, 8].long(),
                "rgbs": self.all_rgbs[idx],
                "rays_train": left_rays[:, :8],
                "ts_train": left_rays[:, 8].long(),
                "rgbs_train_gt": left_rgbs,
                "rays_eval": right_rays[:, :8],
                "ts_eval": right_rays[:, 8].long(),
                "rgbs_eval_gt": right_rgbs,
                "extrinsic": self.extrinsics[idx],
                "intrinsic": self.intrinsics[idx],
                "img_wh": self.img_wh[idx],
                "image_name": self.eval_images[idx],
            }

        # nerf_pl原本评估ckpt用"test_train"这个split，仅新增image_name变量改动
        elif self.split in ["val", "test_train"]:
            sample = {}
            if self.split == "val":
                id_ = self.val_id
            else:
                id_ = self.img_ids_train[idx]
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(
                os.path.join(self.root_dir, "dense/images", self.image_paths[id_])
            ).convert("RGB")
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w // self.img_downscale
                img_h = img_h // self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            sample["rgbs"] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)

            image_name = self.image_paths[id_].split(".")[0]  # 相比pl新增改动，给semaitic_map找npz用

            print(f"【Output】image path id is {id_}")    # id_和dataset[x]这两个数居然不是同一个东西
            print(f"【Output】image name is {image_name}")    # 可正确取到图像文件名

            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.nears[id_] * torch.ones_like(rays_o[:, :1]),
                    self.fars[id_] * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (h*w, 9)
            sample["rays"] = rays
            sample["ts"] = id_ * torch.ones(len(rays), dtype=torch.long)
            if self.with_semantics:
                #todo <修改>强制覆盖semantic_map_path不为None，使用的为defaults中参数
                if self.semantic_map_path is None:
                    self.semantic_map_path = 'semantic_maps'

                # <报错>迁移复现时self.semantic_map_path变为None
                semantic_map = np.load(
                    os.path.join(
                        self.root_dir, f"{self.semantic_map_path}/{image_name}.npz"
                    )
                )["arr_0"]
                semantic_map = cv2.resize(
                    semantic_map,
                    (
                        semantic_map.shape[1] // self.img_downscale,
                        semantic_map.shape[0] // self.img_downscale,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
                rays_mask = semantic_map.reshape(-1, 1)
                sample["semantics"] = torch.from_numpy(rays_mask)
            sample["img_wh"] = torch.LongTensor([img_w, img_h])
            sample["K"] = self.Ks[id_]

        elif self.split == "test":
            sample = {}
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(
                self.test_img_h, self.test_img_w, self.test_K
            )
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    near * torch.ones_like(rays_o[:, :1]),
                    far * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )
            sample["rays"] = rays
            sample["ts"] = self.test_appearance_idx * torch.ones(
                len(rays), dtype=torch.long
            )
            sample["img_wh"] = torch.LongTensor([self.test_img_w, self.test_img_h])
            sample["idx"] = idx
            # sample['K'] = self.Ks[idx]

        return sample
