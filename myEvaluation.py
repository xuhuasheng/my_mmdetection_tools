# =========================================================
# @purpose: evaluate test or inference result with precison and recall
# @date:    2019/12
# @version: v1.0
# @author:  Xu Huasheng
# @github:  https://github.com/xuhuasheng/multitask_thz_detection_framework
# ==========================================================

import os 
import cv2
import progressbar
import numpy as np
from enum import Enum               
from prettytable import PrettyTable

from xml_utils import get_elementTree, get_elements, get_element


class CATEGORY(Enum):
    gun = 1
    phone = 2

# ========================================
WIN_OS = False
# MODEL_NAME = "aug_gray32"
# MODEL_NAME = "aug_pooling77"
# MODEL_NAME = "aug_hog73"
MODEL_NAME = "aug_hog73_gray32_pooling77"
IOU_THRES = 0.5
# ========================================

if WIN_OS:
    GROUNDTRUTH_XMLS_PATH = "F:/IECAS/datasets/SVM_THzDatasets/test2/xmls"
    TESTRESULT_XMLS_PATH = "F:/IECAS/datasets/SVM_THzDatasets/infe_results/{}/infe_result_xmls".format(MODEL_NAME)
else:
    GROUNDTRUTH_XMLS_PATH = "/home/watson/Documents/mask_THzDatasets/test2/xmls"
    # GROUNDTRUTH_XMLS_PATH = "/home/watson/Documents/mask_THzDatasets/val_xmls"
    TESTRESULT_XMLS_PATH = "/home/watson/Documents/mmdet_result/ssd300_coco/xmls"

class DETECTION(Enum):
    bndbox = 1  # bounding box
    rotbox = 2  # oriented bounding box
    seg    = 3  # segmentation

class evaluator:
    def __init__(self):
        self.__gt = 0  # groundtruth:   TP+FN
        self.__rs = 0  # result:        TP+FP
        self.__tp = 0  # true positive: TP
        self.__precision = 0
        self.__recall = 0
        self.__fscore = 0
    def gt_inc(self):
        self.__gt += 1
    def rs_inc(self):
        self.__rs += 1
    def tp_inc(self):
        self.__tp += 1
    def gt_num(self):
        return self.__gt
    def rs_num(self):
        return self.__rs
    def tp_num(self):
        return self.__tp
    def calc_precision(self):
        assert(self.__rs > 0)
        self.__precision = round(self.__tp / self.__rs, 4)
    def get_precision(self):
        return self.__precision
    def calc_recall(self):
        assert(self.__gt > 0)
        self.__recall = round(self.__tp / self.__gt, 4)
    def get_recall(self):
        return self.__recall
    def calc_Fscore(self):
        if self.__precision == 0:
            self.calc_precision()
        if self.__recall == 0:
            self.calc_recall()
        self.__fscore = round((2 * self.__precision * self.__recall) / (self.__precision + self.__recall), 4)
    def get_Fscore(self):
        return self.__fscore

class target:
    def __init__(self, category: CATEGORY):
        self.category = category
        self.bdb = evaluator() # bounding box
        self.obb = evaluator() # oriented bounding box
        self.seg = evaluator() # segmentation
    def categoryName(self):
        return self.category.name
    def eval_bdb(self):
        self.bdb.calc_precision()
        self.bdb.calc_recall()
        self.bdb.calc_Fscore()
    def eval_obb(self):
        self.obb.calc_precision()
        self.obb.calc_recall()
        self.obb.calc_Fscore()
    def eval_seg(self):
        self.seg.calc_precision()
        self.seg.calc_recall()
        self.seg.calc_Fscore()


def evaluate(groundtruth_path, testresult_path, IOU_threshold=0.5):
    """
    Purpose: evaluate test result with groundtruth
    """
    gt_xmls_list = os.listdir(GROUNDTRUTH_XMLS_PATH)
    rs_xmls_list = os.listdir(TESTRESULT_XMLS_PATH)
    
    gt_xmlFile_cnt = 0
    bar = progressbar.ProgressBar(max_value=len(gt_xmls_list)).start()
    # 遍历GT的标注xml文件
    for gt_xmlFileName in gt_xmls_list:
        gt_xmlFile_cnt += 1
        bar.update(gt_xmlFile_cnt)
        # 解析xml元素树, 获得树的根节点
        gt_xml_tree = get_elementTree(os.path.join(GROUNDTRUTH_XMLS_PATH, gt_xmlFileName))
        gt_xml_root = gt_xml_tree.getroot()

        # 获得测试结果的xml文件名
        rs_xmlFileName = '_'.join(gt_xmlFileName.split('.')[0].split('_')[:-1]) + "_result.xml"
        # 解析xml元素树, 获得树的根节点
        rs_xml_tree = get_elementTree(os.path.join(TESTRESULT_XMLS_PATH, rs_xmlFileName)) 
        rs_xml_root = rs_xml_tree.getroot()
        
        # 解析gt和rs的对象列表
        gt_obj_list = get_elements(gt_xml_root, "object")
        rs_det_list = get_elements(rs_xml_root, "detection")

        # 遍历GT中标注object
        for gt_obj in gt_obj_list:
            counter(gt_obj, rs_det_list, IOU_threshold)
               
    bar.finish()
    # 计算评估参数
    eval_and_print(DETECTION.bndbox, IOU_threshold)
    # eval_and_print(DETECTION.rotbox, IOU_threshold)
    # eval_and_print(DETECTION.seg, IOU_threshold)

    print("evaluation finished!")    


def counter(gt_obj, rs_det_list, IOU_threshold):
    """
    Purpose: counter of rs and tp
    Args: 
        gt_obj: from get_elements(gt_xml_root, "object")
        rs_det_list: list  from get_elements(rs_xml_root, "detection")
        IOU_threshold: threshold of IOU
    """
    # 当GT目标是gt_obj_name时
    gt_obj_name = get_element(gt_obj, "name").text
    c = categoryDict[gt_obj_name]
    c.bdb.gt_inc()
    c.obb.gt_inc()
    c.seg.gt_inc()
    # 遍历测试结果的检测标注
    for rs_det in rs_det_list:
        if get_element(rs_det, "name").text == gt_obj_name:
            c.bdb.rs_inc()    
            c.obb.rs_inc()
            c.seg.rs_inc()
            # ============= bdb =============
            if get_IoU(gt_obj, rs_det, DETECTION.bndbox) >= IOU_threshold:
                c.bdb.tp_inc() 
            # # # ============= obb =============
            # if get_IoU(gt_obj, rs_det, DETECTION.rotbox) >= IOU_threshold:
            #     c.obb.tp_inc() 
            # # # ============= seg =============
            # if get_IoU(gt_obj, rs_det, DETECTION.seg) >= IOU_threshold:
            #     c.seg.tp_inc() 


def eval_and_print(detection, IOU_threshold):
    """
    Purpose: calculate args of evaluation and print
    Args: 
        detection: bndbox, rotbox, seg
    """
    if detection is DETECTION.bndbox:
        for c in categoryDict.values():
            c.eval_bdb()
        tb = PrettyTable()
        tb.field_names = ["class", "GT", "DET", "TP", "recall", "precision", "F-score"]
        for c in categoryDict.values():
            tb.add_row([c.categoryName(), c.bdb.gt_num(), c.bdb.rs_num(), c.bdb.tp_num(), c.bdb.get_recall(), c.bdb.get_precision(), c.bdb.get_Fscore()])
        tb.add_row(["mean", "", "", "", calc_mean_recall(detection), calc_mean_precision(detection), calc_mean_fscore(detection)])
        print("\n================== bndbox @IoU = {} ===================".format(IOU_threshold))
        print(tb)
        bdb_mP_list.append(calc_mean_precision(detection))
        bdb_mR_list.append(calc_mean_recall(detection))
        bdb_mF_list.append(calc_mean_fscore(detection))
    elif detection is DETECTION.rotbox:
        for c in categoryDict.values():
            c.eval_obb()
        tb = PrettyTable()
        tb.field_names = ["class", "GT", "DET", "TP", "recall", "precision", "F-score"]
        for c in categoryDict.values():
            tb.add_row([c.categoryName(), c.obb.gt_num(), c.obb.rs_num(), c.obb.tp_num(), c.obb.get_recall(), c.obb.get_precision(), c.obb.get_Fscore()])
        tb.add_row(["mean", "", "", "", calc_mean_recall(detection), calc_mean_precision(detection), calc_mean_fscore(detection)])
        print("\n================== rotbox @IoU = {} ==================".format(IOU_threshold))
        print(tb)
        obb_mP_list.append(calc_mean_precision(detection))
        obb_mR_list.append(calc_mean_recall(detection))
        obb_mF_list.append(calc_mean_fscore(detection))
    elif detection is DETECTION.seg:
        for c in categoryDict.values():
            c.eval_seg()
        tb = PrettyTable()
        tb.field_names = ["class", "GT", "DET", "TP", "recall", "precision", "F-score"]
        for c in categoryDict.values():
            tb.add_row([c.categoryName(), c.seg.gt_num(), c.seg.rs_num(), c.seg.tp_num(), c.seg.get_recall(), c.seg.get_precision(), c.seg.get_Fscore()])
        tb.add_row(["mean", "", "", "", calc_mean_recall(detection), calc_mean_precision(detection), calc_mean_fscore(detection)])
        print("\n================== seg @IoU = {} ==================".format(IOU_threshold))
        print(tb)
        seg_mP_list.append(calc_mean_precision(detection))
        seg_mR_list.append(calc_mean_recall(detection))
        seg_mF_list.append(calc_mean_fscore(detection))
    else:
        raise Exception("No Such Detection!")


def calc_mean_recall(detection):
    """
    Purpose: calculate mean recall
    Args: 
        category: list of category: class target
        detection: bndbox, rotbox, seg
    """
    mR = 0
    if detection is DETECTION.bndbox:
        for c in categoryDict.values():
            if c.bdb.get_recall() == 0:
                c.bdb.calc_recall()
            mR += c.bdb.get_recall()
    elif detection is DETECTION.rotbox:
        for c in categoryDict.values():
            if c.obb.get_recall() == 0:
                c.obb.calc_recall()
            mR += c.obb.get_recall()
    elif detection is DETECTION.seg:
        for c in categoryDict.values():
            if c.seg.get_recall() == 0:
                c.seg.calc_recall()
            mR += c.seg.get_recall()
    else:
        raise Exception("No Such Detection!")
    mR = round(mR/len(categoryDict), 4)
    return mR


def calc_mean_precision(detection):
    """
    Purpose: calculate mean precision 
    Args: 
        category: list of category: class target
        detection: bndbox, rotbox, seg
    """
    mP = 0
    if detection is DETECTION.bndbox:
        for c in categoryDict.values():
            if c.bdb.get_precision() == 0:
                c.bdb.calc_precision()
            mP += c.bdb.get_precision()
    elif detection is DETECTION.rotbox:
        for c in categoryDict.values():
            if c.obb.get_precision() == 0:
                c.obb.calc_precision()
            mP += c.obb.get_precision()
    elif detection is DETECTION.seg:
        for c in categoryDict.values():
            if c.seg.get_precision() == 0:
                c.seg.calc_precision()
            mP += c.seg.get_precision()
    else:
        raise Exception("No Such Detection!")
    mP = round(mP/len(categoryDict), 4)
    return mP


def calc_mean_fscore(detection):
    """
    Purpose: calculate mean fscore 
    Args: 
        category: list of category: class target
        detection: bndbox, rotbox, seg
    """
    mf = 0
    if detection is DETECTION.bndbox:
        for c in categoryDict.values():
            if c.bdb.get_Fscore() == 0:
                c.bdb.calc_Fscore()
            mf += c.bdb.get_Fscore()
    elif detection is DETECTION.rotbox:
        for c in categoryDict.values():
            if c.obb.get_Fscore() == 0:
                c.obb.calc_Fscore()
            mf += c.obb.get_Fscore()
    elif detection is DETECTION.seg:
        for c in categoryDict.values():
            if c.seg.get_Fscore() == 0:
                c.seg.calc_Fscore()
            mf += c.seg.get_Fscore()
    else:
        raise Exception("No Such Detection!")
    mf = round(mf/len(categoryDict), 4)
    return mf


def get_IoU(gt_obj, rs_det, detection):
    """
    Purpose: return a IoU of groundtruth and result  
    Args: 
        gt_obj: element in groundtruth xml 
        rs_det: element in result xml 
        detection: bndbox, rotbox, seg
    Returns: 
        iou: the IOU of groundtruth and result 
    """
    if detection is DETECTION.bndbox:
        # 获得GT的gun的bbox
        gt_bdb = get_element(gt_obj, "bndbox")
        gt_bdb_xmin = int(get_element(gt_bdb, "xmin").text)
        gt_bdb_ymin = int(get_element(gt_bdb, "ymin").text)
        gt_bdb_xmax = int(get_element(gt_bdb, "xmax").text)
        gt_bdb_ymax = int(get_element(gt_bdb, "ymax").text)
        gt_bbox = xyxy2xywh([gt_bdb_xmin, gt_bdb_ymin, gt_bdb_xmax, gt_bdb_ymax])
        # 获得测试结果的gun的bbox
        rs_bdb = get_element(rs_det, "bndbox")
        rs_bdb_x = int(get_element(rs_bdb, "x").text)
        rs_bdb_y = int(get_element(rs_bdb, "y").text)
        rs_bdb_w = int(get_element(rs_bdb, "w").text)
        rs_bdb_h = int(get_element(rs_bdb, "h").text)
        rs_bbox = [rs_bdb_x, rs_bdb_y, rs_bdb_w, rs_bdb_h]
        # 获得GT和res的交并比
        iou = get_IoU_rect(gt_bbox, rs_bbox)
    elif detection is DETECTION.rotbox:
        gt_obb_poly = text2polygon(get_element(gt_obj, "rotbox").text)
        rs_obb_poly = text2polygon(get_element(rs_det, "rotationbox").text)
        iou = get_IoU_polygon(gt_obb_poly, rs_obb_poly)
    elif detection is DETECTION.seg:
        gt_seg_poly = text2polygon(get_element(gt_obj, "segmentation").text)
        rs_seg_poly = text2polygon(get_element(rs_det, "segmentation").text)
        iou = get_IoU_polygon(gt_seg_poly, rs_seg_poly)
    else:
        raise Exception("Unknow detection!")
    return iou


def get_IoU_rect(bbox1, bbox2):
    """
    Purpose: return a IoU of bbox1 and bbox2  
    Args: bbox1 and bbox2 is [x, y, w, h]
    Returns: the IOU of bbox1 and bbox2
    """
    # bbox = [x,y,w,h]
    # Calculate the x-y co-ordinates of the rectangles
    x1_tl = bbox1[0]
    x2_tl = bbox2[0]
    x1_br = bbox1[0] + bbox1[2]
    x2_br = bbox2[0] + bbox2[2]
    y1_tl = bbox1[1]
    y2_tl = bbox2[1]
    y1_br = bbox1[1] + bbox1[3]
    y2_br = bbox2[1] + bbox2[3]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = bbox1[2] * bbox2[3]
    area_2 = bbox2[2] * bbox2[3]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


def get_IoU_polygon(polygon1, polygon2):
    """
    Purpose: get IoU of two polygons  
    Args: polygon: array([[x0,y0], [x1,y1], [x2,y2], [x3,y3], ...], dtype=np.int)
    Returns: the IOU of polygon1 and polygons2
    """
    shape = (391, 159)    # default
    img1 = cv2.fillConvexPoly(np.zeros(shape, np.uint8), polygon1, 100) # 填充为100
    img2 = cv2.fillConvexPoly(np.zeros(shape, np.uint8), polygon2, 155) # 填充为155
    img = img1 + img2
    ret, overlapImg = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY) # 过滤出重叠部分(像素值为255)
    overlapArea = np.sum(np.greater(overlapImg, 0)) # 求取两个多边形交叠区域面积
    area1 = cv2.contourArea(polygon1)
    area2 = cv2.contourArea(polygon2)
    totalArea = area1 + area2 - overlapArea
    return overlapArea / float(totalArea)


def xyxy2xywh(bbox_xyxy):
    """
    Args:
        bbox_xyxy: [xim, yxim, xmax, ymax]
    Ruturns:
        bbox_xywh: [x, y, w, h]
    """
    xim, yxim, xmax, ymax = bbox_xyxy
    return [xim, yxim, xmax-xim, ymax-yxim]


def seg_text2listxy(text):
    """
    Purpose: parse x, y coordinates from text of seg (segmentation) annotation in xml
    Args: 
        text: text of seg (segmentation) annotation in xml, "[x0,y0, x1,y1, x2,y2, x3,y3, ...]"
    Returns:  lists of storing x y coordinates, 
        x: [x0, x1, x2, x3, ...]
        y: [y0, y1, y2, y3, ...]
    """
    strList = text[1:-1].split(", ")
    x = list(map(int, strList[::2]))
    y = list(map(int, strList[1::2]))
    return x, y


def obb_text2listxy(text):
    """
    Purpose: parse x, y coordinates from text of obb(rotbox) annotation in xml
    Args: 
        text: text of obb(rotbox) annotation in xml, "[[x0,y0], [x1,y1], [x2,y2], [x3,y3]]"
    Returns:  lists of storing x y coordinates, 
        x: [x0, x1, x2, x3]
        y: [y0, y1, y2, y3]
    """
    strList = list(text)[1:-1] # 去掉首尾的'[', ']'
    # list of storing x y coordinates
    x = []
    y = []
    # double pointers
    i = 0
    j = 0
    while i != len(strList):
        if strList[i] == '[':
            x_ = ''
            j = i+1
            while strList[j] != ',':
                x_ += strList[j]
                j += 1
            x.append(int(x_))
            i = j
        if strList[i] == ' ' and strList[i+1] != '[':
            y_ = ''
            j = i+1
            while strList[j] != ']':
                y_ += strList[j]
                j += 1
            y.append(int(y_))
            i = j
        i += 1
    return x, y


def text2polygon(text):
    """
    Purpose: parse text of annotation in xml to polygon format
    Args: 
        text of obb or seg: "[[x0,y0], [x1,y1], [x2,y2], [x3,y3]]" or "[x0,y0, x1,y1, x2,y2, x3,y3, ...]"
    Returns:  
        polygon: array([[x0,y0], [x1,y1], [x2,y2], [x3,y3], ...], dtype=np.int)
    """
    if text[0] == '[' and text[1] == '[':
        listx, listy = obb_text2listxy(text)
    else:
        listx, listy = seg_text2listxy(text)
    npts = len(listx)
    polygon = np.zeros(shape=(npts, 2), dtype=np.int)
    for i in range(npts):
        polygon[i, 0] = listx[i]
        polygon[i, 1] = listy[i]
    return polygon
 
 
if __name__ == "__main__":
    # =========================================================
    # # 目标对象统计器
    # gun = target(CATEGORY.gun)
    # phone = target(CATEGORY.phone)
    # # dict of target classes
    # categoryDict = {"gun": gun, "phone": phone}
    # evaluate(GROUNDTRUTH_XMLS_PATH, TESTRESULT_XMLS_PATH, IOU_threshold=IOU_THRES)

    # ==========================================================
    iou_init = 0.3
    iou_list = [round(iou_init + i*0.05, 2) for i in range(10)]

    bdb_mP_list = []
    bdb_mR_list = []
    bdb_mF_list = []

    obb_mP_list = []
    obb_mR_list = []
    obb_mF_list = []

    seg_mP_list = []
    seg_mR_list = []
    seg_mF_list = []
    for i, iou in enumerate(iou_list):
        # 目标对象统计器
        gun = target(CATEGORY.gun)
        phone = target(CATEGORY.phone)
        # dict of target classes
        categoryDict = {"gun": gun, "phone": phone}
        evaluate(GROUNDTRUTH_XMLS_PATH, TESTRESULT_XMLS_PATH, IOU_threshold=iou)

    print("bdb_mP_list: {}".format(bdb_mP_list))
    print("bdb_mR_list: {}".format(bdb_mR_list))
    print("bdb_mF_list: {}".format(bdb_mF_list))

    print("obb_mP_list: {}".format(obb_mP_list))
    print("obb_mR_list: {}".format(obb_mR_list))
    print("obb_mF_list: {}".format(obb_mF_list))

    print("seg_mP_list: {}".format(seg_mP_list))
    print("seg_mR_list: {}".format(seg_mR_list))
    print("seg_mF_list: {}".format(seg_mF_list))