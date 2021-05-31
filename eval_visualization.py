import os
import mmcv
TEST_RESULT_ROOT = "/home/watson/anaconda3/envs/pytorch14/mmdetection/test_result"
PKL_FILE = "mask_rcnn_r50_fpn_1x_coco_senet/latest.pkl"

result_file = os.path.join(TEST_RESULT_ROOT, PKL_FILE)
test_results = mmcv.load(result_file)
print(test_results)