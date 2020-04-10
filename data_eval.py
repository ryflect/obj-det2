import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line[:-2] for line in lines]
gt_split_file = "/mnt/nfs/scratch1/pmallya/nusc_kitti/val/nusc_val.txt" 
val_image_ids = _read_imageset_file(gt_split_file)
# print("Val Image IDs: ", val_image_ids)
det_path = "/mnt/nfs/scratch1/pmallya/nusc_kitti/val/infer_2/"
dt_annos = kitti.get_label_annos(det_path, val_image_ids)
gt_path = "/mnt/nfs/scratch1/pmallya/nusc_kitti/val/label_2/"

# gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
print(dt_annos)
print(gt_annos)
print(get_official_eval_result(gt_annos, dt_annos, 0)) 
print(get_coco_eval_result(gt_annos, dt_annos, 0)) 