import trimesh
import numpy as np
from scipy.spatial import cKDTree
from pipelines.utils.libmesh import check_mesh_contains

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
              points_iou, occ_tgt):
    ''' Evaluates a mesh.

    Args:
        mesh (trimesh): mesh which should be evaluated
        pointcloud_tgt (numpy array): target point cloud
        normals_tgt (numpy array): target normals
        points_iou (numpy_array): points tensor for IoU evaluation
        occ_tgt (numpy_array): GT occupancy values for IoU points
    '''
    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        pointcloud, idx = mesh.sample(self.n_points, return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh.face_normals[idx]
    else:
        pointcloud = np.empty((0, 3))
        normals = np.empty((0, 3))

    out_dict = self.eval_pointcloud(
        pointcloud, pointcloud_tgt, normals, normals_tgt)

    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        occ = check_mesh_contains(mesh, points_iou)
        out_dict['iou'] = compute_iou(occ, occ_tgt)
    else:
        out_dict['iou'] = 0.

    return out_dict

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src, n_jobs=32)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def eval_pointcloud(pointcloud, pointcloud_tgt, normals=None, normals_tgt=None):
    ''' Evaluates a point cloud.

    Args:
        pointcloud (numpy array): predicted point cloud
        pointcloud_tgt (numpy array): target point cloud
        normals (numpy array): predicted normals
        normals_tgt (numpy array): target normals
    '''
    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    dist_backward, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud, normals
    )
    dist_backward2 = dist_backward**2

    completeness = dist_backward.mean()
    completeness2 = dist_backward2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    dist_forward, accuracy_normals = distance_p2p(
        pointcloud, normals, pointcloud_tgt, normals_tgt
    )
    dist_forward2 = dist_forward**2

    accuracy = dist_forward.mean()
    accuracy2 = dist_forward2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)

    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer-L2': chamferL2,
        'chamfer-L1': chamferL1,
    }

    # F-score
    # percentage_list_1 = np.arange(0.0001, 0.001, 0.0001).astype(np.float32)
    # percentage_list_2 = np.arange(0.001, 0.01, 0.001).astype(np.float32)
    # percentage_list_3 = np.arange(0.01, 0.11, 0.01).astype(np.float32)
    # thres_percentage_list = np.concatenate([percentage_list_1, percentage_list_2, percentage_list_3], axis=0)
    # thres_percentage_list = np.sort(thres_percentage_list)
    # xmax = pointcloud_tgt.max(axis=0)
    # xmin = pointcloud_tgt.min(axis=0)
    # bbox_length = np.linalg.norm(xmax - xmin)
    # threshold_list = bbox_length * thres_percentage_list
    threshold_list = np.array([0.005]).astype(np.float32)
    for i in range(threshold_list.shape[0]):
        threshold = threshold_list[i]

        pre_sum_val = np.sum(np.less(dist_forward, threshold))
        rec_sum_val = np.sum(np.less(dist_backward, threshold))
        fprecision = pre_sum_val / dist_forward.shape[0]
        frecall = rec_sum_val / dist_backward.shape[0]
        fscore = 2 * (fprecision * frecall) / (fprecision + frecall + 1e-6)
        out_dict['f_score_{:.4}'.format(threshold)] = fscore
        out_dict['precision_{:.4}'.format(threshold)] = fprecision
        out_dict['rescall_{:.4}'.format(threshold)] = frecall

    return out_dict