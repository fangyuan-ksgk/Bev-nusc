from train_nuscenes import *
from nuscenesdataset import * 
from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix


def unpack_rot_trans(cam_egopose, inverse=False):
    cam2ego = np.zeros((4,4))
    cam2ego[:3,:3] = Quaternion(cam_egopose['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.array(cam_egopose['translation'])
    cam2ego[3,3] = 1.
    if inverse:
        ego2cam = np.linalg.inv(cam2ego)
        return ego2cam
    return cam2ego

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
class Nusc_bev(torch.utils.data.Dataset):
# class Nusc_bev:
    def __init__(self, nusc, is_train, data_aug_conf, centroid=None, bounds=None, res_3d=None,
                  nsweeps=1, seqlen=1, refcam_id=1):
        # super().__init__()
        # print('Initalizing Nusc Dataset')
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        # self.grid_conf = grid_conf
        self.nsweeps = nsweeps
        self.res_3d = res_3d
        self.bounds = bounds
        self.centroid = centroid

        self.seqlen = seqlen
        self.refcam_id = refcam_id
        self.dataroot = self.nusc.dataroot
        self.scenes = self.get_scenes()
        self.prepro()
        
        # Configuration Params: mesh size & long limit & lat limit
        self.mesh_xy = [0.1,0.1]
        # long & lat in order
        self.min_xy = [-50., -35.]
        self.max_xy = [90., 35.]
        
    def __new__(cls, nusc, is_train, data_aug_conf, centroid=None, bounds=None, res_3d=None,
                  nsweeps=1, seqlen=1, refcam_id=1):
        print("Creating Instance")
        instance = super(Nusc_bev, cls).__new__(cls)
        return instance
        
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]
        scenes = create_splits_scenes()[split]
        return scenes
    
    def prepro(self):
        samples = [samp for samp in self.nusc.sample]
        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        return samples
    
    # Get 3D BBox under Ego vehicle cooridnate -- timestamp same as the reference camera (FRONT CAM)
    def get_bev_bbox(self, idx):
        samp = self.nusc.sample[idx]
        self.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        imgs = []
        img_metas = []
        # 3D Box objects under ego vehicle coordinate
        boxes_ego = []
        extrins = []
        intrins = []

        # Lidar Sensor Information
        lid_rec = self.nusc.get('sample_data', samp['data']['LIDAR_TOP'])
        lid_egopose = self.nusc.get('ego_pose', lid_rec['ego_pose_token'])
        # print('Lidar ', np.array(lid_egopose['translation']))
        lid_calib = self.nusc.get('calibrated_sensor', lid_rec['calibrated_sensor_token'])

        # LIDAR 3DBBox
        pc_path, boxes_lid, _ = self.nusc.get_sample_data(samp['data']['LIDAR_TOP'])

        # Reference Camera Information : Front Camera Chosen
        refcam = 'CAM_FRONT'
        refcam_rec = self.nusc.get('sample_data', samp['data'][refcam])
        refcam_egopose = self.nusc.get('ego_pose', refcam_rec['ego_pose_token'])
        refcam_calib = self.nusc.get('calibrated_sensor', refcam_rec['calibrated_sensor_token'])
        glob2ego_refcam = unpack_rot_trans(refcam_egopose, inverse=True)

        # 3D Bbox collection: Convert from Lidar coordiante towards Ego vehicle -- refcam timestamp
        for box in boxes_lid:
            # lid2ego_lid
            box.rotate(Quaternion(lid_calib['rotation']))
            box.translate(np.array(lid_calib['translation']))
            #ego_lid2glob
            box.rotate(Quaternion(lid_egopose['rotation']))
            box.translate(np.array(lid_egopose['translation']))
            #glob2ego_refcam
            box.translate(-np.array(refcam_egopose['translation']))
            box.rotate(Quaternion(refcam_egopose['rotation']).inverse)
            boxes_ego.append(box)

        # Images & Image Metas collection
        for cam in cams:
            cam_rec = self.nusc.get('sample_data', samp['data'][cam])
            # For all cameras, camera pose are the same
            cam_egopose = self.nusc.get('ego_pose', cam_rec['ego_pose_token'])
            cam_calib = self.nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
            # INTRINSIC
            img_path, boxes_cam, camera_intrinsic = self.nusc.get_sample_data(samp['data'][cam])
            intrins.append(camera_intrinsic)
            # (print(cam, np.array(cam_egopose['translation'])))
            imgs.append(Image.open(img_path))
            # extrinsic, cam2ego (ego-front-cam, we ignore the different between camers)
            # so the front-camera would be the most accurate camera on which we provide our estimate
            # cam2ego(cam) -> ego(cam)2glob -> glob2ego(refcam) = cam2ego (refcam) = extrinsic
            cam2ego_cam = unpack_rot_trans(cam_calib)
            ego_cam2glob = unpack_rot_trans(cam_egopose)
            cam2ego_refcam = glob2ego_refcam @ ego_cam2glob @ cam2ego_cam
            extrins.append(cam2ego_refcam)

        return imgs, img_metas, boxes_ego, intrins, extrins
    
    def visualize_bev_box(self, idx, refcam='CAM_FRONT'):
        # bev box obtained 
        imgs, img_metas, boxes_ego, intrins, extrins = self.get_bev_bbox(idx)
        
        samp = self.nusc.sample[idx]
        refcam_rec = self.nusc.get('sample_data', samp['data'][refcam])
        refcam_calib = self.nusc.get('calibrated_sensor', refcam_rec['calibrated_sensor_token'])
        
        # visualize on refcam : Front Camera
        img_path, _, camera_intrinsic = self.nusc.get_sample_data(samp['data'][refcam])
        # read image
        img = Image.open(img_path)
        # Init axes
        _, ax = plt.subplots(1, 1, figsize=(9, 16))
        # Plot image
        ax.imshow(img)

        # This should now resides under camera coordinate
        for box in boxes_ego:
            box.translate(-np.array(refcam_calib['translation']))
            box.rotate(Quaternion(refcam_calib['rotation']).inverse)
            if box.center[2]<=1 or abs(box.center[0])>=10 or not(box.name.startswith('human') or box.name.startswith('vehicle')):
                continue
            c = np.array(self.nusc.explorer.get_color(box.name)) / 255.0
            box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))
    
    def get_bev_plot(self, idx, refcam='CAM_FRONT'):
        samp = self.nusc.sample[0]
        # EXTRINSIC
        lid_rec = self.nusc.get('sample_data', samp['data']['LIDAR_TOP'])
        lid_egopose = self.nusc.get('ego_pose', lid_rec['ego_pose_token'])
        lid_calib = self.nusc.get('calibrated_sensor', lid_rec['calibrated_sensor_token'])
        cam_rec = self.nusc.get('sample_data', samp['data'][refcam])
        cam_egopose = self.nusc.get('ego_pose', cam_rec['ego_pose_token'])
        cam_calib = self.nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])

        # get bev 3D bbox
        imgs, img_metas, boxes_ego, intrins, extrins = self.get_bev_bbox(idx)
        # BEV projection & visulization (For training purpose, we need a 1-dim vector here as target supervision)
        img = np.zeros(((np.array(self.max_xy) - np.array(self.min_xy)) / np.array(self.mesh_xy)).astype(int).tolist())

        for ii, box in enumerate(boxes_ego):
            box.translate(-np.array(cam_calib['translation']))
            box.rotate(Quaternion(cam_calib['rotation']).inverse)
            # NuScenes filter
            if 'vehicle' not in box.name:
                continue
            if discard_invisible and int(inst['visibility_token']) == 1:
                # filter invisible vehicles
                continue
            # break
            pts = box.bottom_corners()[::2].T
            # print('Bottom points under FrontCamera coordinate: ', pts)
            # Front Cmaera coordiante longi for z, lati for x, so we reverse the mesh_xy oreer here
            pts_nr = (pts - self.min_xy[::-1]) / self.mesh_xy[::-1]
            pts = np.round(pts_nr).astype(np.int32)
            # print('No-rounded points: ', pts_nr)
            # print('Rounded Points: ', pts)
            img = cv2.fillPoly(img, [pts], 10.0)
        return img
    
    def visualize_bev_plot(self, idx, refcam='CAM_FRON'):
        img = self.get_bev_plot(idx, refcam)
        w_long, w_lat = img.shape
        lat_tick = np.arange(0, w_lat+int(5/self.mesh_xy[1]), int(5/self.mesh_xy[1]))
        lat_map = lat_tick * self.mesh_xy[1] + self.min_xy[1]
        long_tick = np.arange(0, w_long+int(5/self.mesh_xy[0]), int(5/self.mesh_xy[0]))
        long_map = long_tick * self.mesh_xy[0] + self.min_xy[0]
        cpos = np.array([0.,0.])
        # cpos_conv = ((cpos  - np.array(min_xy))/(np.array(max_xy)-np.array(min_xy)))/np.array(mesh_xy).tolist()
        cpos_conv = (cpos  - np.array(self.min_xy)) / np.array(self.mesh_xy)

        plt.imshow(img)
        plt.scatter(cpos_conv[1],cpos_conv[0],marker='*',c='red')
        plt.xticks(lat_tick.tolist(), lat_map.tolist(), rotation='vertical')
        plt.yticks(long_tick.tolist(), long_map.tolist())
        plt.ylim(0., w_long)
        plt.show()

            
    def visualize_bev_box_specific(self, idx, box_indices, refcam='CAM_FRONT'):
        # bev box obtained 
        imgs, img_metas, boxes_ego, intrins, extrins = self.get_bev_bbox(idx)
        
        samp = nusc.sample[idx]
        refcam_rec = nusc.get('sample_data', samp['data'][refcam])
        refcam_calib = nusc.get('calibrated_sensor', refcam_rec['calibrated_sensor_token'])
        
        # visualize on refcam : Front Camera
        img_path, _, camera_intrinsic = nusc.get_sample_data(samp['data'][refcam])
        # read image
        img = Image.open(img_path)
        # Init axes
        _, ax = plt.subplots(1, 1, figsize=(9, 16))
        # Plot image
        ax.imshow(img)

        # This should now resides under camera coordinate
        for box in [boxes_ego[box_idx] for box_idx in box_indices]:
            box.translate(-np.array(refcam_calib['translation']))
            box.rotate(Quaternion(refcam_calib['rotation']).inverse)
            pts = box.bottom_corners()[::2].T
            print('Bottom points under FrontCamera coordinate: ', pts)
            
            if box.center[2]<=1 or abs(box.center[0])>=10 or not(box.name.startswith('human') or box.name.startswith('vehicle')):
                continue
            c = np.array(nusc.explorer.get_color(box.name)) / 255.0
            box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))