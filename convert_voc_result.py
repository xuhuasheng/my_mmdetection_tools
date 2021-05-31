import os 
import mmcv
import cv2
import numpy as np 
from math import atan2, cos, sin, sqrt, pi
from enum import Enum
from xml_utils import create_element, add_childElement, create_elementTree, formatXml
class OBB_MODE(Enum):
    # method of calculating rotbox 
    PCA = 1
    CONVEXHULL = 2


def resolve_result(result, class_names, score_thr=0.3):
    # 拆包
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    
    # 整理
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)

    # 过虑
    
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    if segm_result is not None:
        assert len(segms) == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        scores = scores[inds]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segm_result is not None:
            segms = [seg for i, seg in zip(inds, segms) if i]
        else:
            segms = [None for i in inds if i]
        

    # 解析
    detections = []
    for score, bbox, segm, label in zip(scores, bboxes, segms, labels):
        predName = class_names[label] if class_names is not None else f'cls {label}'

        bbox_int = bbox.astype(np.int32) # [x1, y1, x3, y3]
        bndbox = [bbox_int[0], bbox_int[1], bbox_int[2]-bbox_int[0], bbox_int[3]-bbox_int[1]]

        if segm_result is not None:
            mask = segm.astype(np.uint8)
            maxContour = findMaxContour(mask, visualize=False)
            seg = get_segmentation(maxContour, 0, 0, subsampleRate=None, coordinateNum=20)
        else:
            seg = []
            
        rotbox = []
        # rotbox = get_rotbox(maxContour, 0, 0, method=OBB_MODE.CONVEXHULL)
        # draw_rotbox(segm.astype(np.uint8), rotbox, color=(0,0,200), thickness=1, rotbox_mode=OBB_MODE.CONVEXHULL)

        detections.append({"class": predName, "score": score, "bndbox": bndbox, "segmentation": seg, "rotbox": rotbox})

    return detections



def store_voc_result(img_fullname, detections_list, xml_fullName):
    """
    Purpose: generate and write a xml to store detection results
    Args:
        img_fullname: the full path to image detected
        detections_list[dict]: the list of detection dict
        xml_fullName: the full path to store xml file 
    """
    # root == results
    _results = create_element("results", {}, None)   
    # results->imgFUllName
    _imgFullName = create_element("imgFullName", {}, img_fullname)
    add_childElement(_results, _imgFullName)
    # results->detection
    for det_dict in detections_list:
        # get argments from dict
        class_name = det_dict["class"]
        x, y, w, h = det_dict["bndbox"]
        seg = det_dict["segmentation"]
        rotbox = det_dict["rotbox"]

        _detection = create_element("detection", {}, None)
        # results->detection->name
        _name =create_element("name", {}, class_name)
        add_childElement(_detection, _name)
        # results->detection->bndbox->x,y,w,h
        _bndbox = create_element("bndbox", {}, None)
        _x = create_element("x", {}, str(x))
        _y = create_element("y", {}, str(y))
        _w = create_element("w", {}, str(w))
        _h = create_element("h", {}, str(h))
        add_childElement(_bndbox, _x)
        add_childElement(_bndbox, _y)
        add_childElement(_bndbox, _w)
        add_childElement(_bndbox, _h)
        add_childElement(_detection, _bndbox)
        # results->detection->segmentaion
        _segmentation = create_element("segmentation", {}, str(seg))
        add_childElement(_detection, _segmentation)
        # results->detection->rotationbox
        _rotbox = create_element("rotationbox", {}, str(rotbox))
        add_childElement(_detection, _rotbox)
        add_childElement(_results, _detection)

    formatXml(_results, '\t', '\n')      # 美化xml格式
    tree = create_elementTree(_results)  # 以根元素创建elementtree对象
    tree.write(xml_fullName)             # 将构建的XML文档写入文件


def findMaxContour(binary_image, visualize=False):
    """
    Purpose: find all contours in the binarized image and return the maximum
    Args:
        binary_image: a binarized image  
    Ruturns:
        maxContour: the maximum contour surrounding the largest area    
    """
    # 寻找所有轮廓
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    # 轮廓面积
    contour_Area = 0
    # 遍历所有轮廓 寻找面积最大的轮廓
    maxContour = contours[0]
    for i in range(len(contours)):
        contour_Area_temp = cv2.contourArea(contours[i])
        if contour_Area_temp > contour_Area:
            contour_Area = contour_Area_temp
            maxContour = contours[i] # 最大的轮廓
    if visualize:
        img_maxContour = binary_image.copy()
        img_maxContour = cv2.merge([img_maxContour, img_maxContour, img_maxContour])
        # cv2.drawContours(img_maxContour, contours, -1, (0,255,0), 2) # draw all contours with green
        cv2.drawContours(img_maxContour, [maxContour], 0, (0,0,255), 2)  # draw max contour with red
        cv2.imshow("img_maxContour", img_maxContour)
        cv2.waitKey(0)
    return maxContour


def get_segmentation(contour, window_xmin, window_ymin, subsampleRate=None, coordinateNum=20):
    """
    Purpose: Get the coordinates of the contour points and perform interval sampling
    Args:
        contour: a contour to be performed, which is a 3-d array [ [[x0,y0]], [[x1,y1]], ... ].shape() = (n-row, 1, 2)
        window_xmin: x coordinate of the top-left point of the sliding window, 
                     used for the calculation of global coordinates
        window_ymin: y coordinate of the top-left point of the sliding window, 
                     used for the calculation of global coordinates
        subsampleRate: Interval sampling rate for contour point coordinates
        coordinateNum: Number of contour points after interval sampling
    Ruturns:
        seg: Global coordinates of contour points, which is a list [x0,y0, x1,y1, x2,y2, ...]  
    """
    # 轮廓点下采样
    new_contour = contour_subsample(contour, subsampleRate=None, subsampleNum=20)
    # 将轮廓点的3维array转化为一维list
    seg = []
    for pt in new_contour:
        seg.append(pt[0][0]) # x
        seg.append(pt[0][1]) # y
    # 把目标轮廓坐标转化为全局坐标
    for i in range(len(seg)):
        if i%2 == 0: # x
            seg[i] = seg[i] + window_xmin
        else: # y
            seg[i] = seg[i] + window_ymin
    return seg


def contour_subsample(contour, subsampleRate=None, subsampleNum=20):
    """
    Purpose: subsample contour, reduce the number of contour points
    Args:
        contour: contour to be performed
        subsampleRate: Interval sampling rate for contour point coordinates
        subsampleNum: Number of contour points after interval sampling             
    Ruturns: 
        simplified_contour: contour after subsampling
    """
    contour_ptNum = contour.shape[0]
    if contour_ptNum <= subsampleNum:
        subsampleNum = contour_ptNum
    if subsampleRate is None:
        subsampleRate = int(contour_ptNum / subsampleNum)
    simplified_contour = np.empty((subsampleNum, 1, 2), dtype=np.int32)
    i = 0
    for j in range(subsampleNum):
        simplified_contour[j, 0, :] = contour[i, 0, :]
        i = i + subsampleRate
    return simplified_contour


# =====================================
#              OBB Branch
# =====================================
def get_rotbox(contour, window_xmin, window_ymin, method=OBB_MODE.PCA): 
    """
    Purpose: get min area box of contour with rotation angle
    Args:
        contour: contour to be performed
        window_xmin: x coordinate of the top-left point of the sliding window, 
                     used for the calculation of global coordinates
        window_ymin: y coordinate of the top-left point of the sliding window, 
                     used for the calculation of global coordinates
        method: OBB_MODE.PCA or OBB_MODE.CONVEXHULL             
    Ruturns: 
        rotbox: vertexes coordinates list: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    """
    if method is OBB_MODE.PCA:
        center, angle, _, _ = get_orientation(contour)
        upright_contour = rotate_contour(contour, center, angle)
        upright_bndRect = cv2.boundingRect(upright_contour) # bndbox = (x, y, w, h) 
        upright_bndRectPt = [[round(upright_bndRect[0]), round(upright_bndRect[1])], # p0=(x,y)
                            [round(upright_bndRect[0])+round(upright_bndRect[2]), round(upright_bndRect[1])], # p1=(x+w,y)
                            [round(upright_bndRect[0])+round(upright_bndRect[2]), round(upright_bndRect[1])+round(upright_bndRect[3])], # p2=(x+w,y+h)
                            [round(upright_bndRect[0]), round(upright_bndRect[1])+round(upright_bndRect[3])]] # p3=(x,y+h)
        rotbox = []
        for pt in upright_bndRectPt:
            x_, y_ = rotate_point(tuple(pt), center, -angle)
            rotbox.append([int(x_ + window_xmin), int(y_ + window_ymin)])
    elif method is OBB_MODE.CONVEXHULL:
        minRect = cv2.minAreaRect(contour)    # minRect:tuple, minRect[0] = (center_x, center_y), minRect[1] = (w, h), minRect[2] = angle((-90,0])
        minRect_ = ((round(minRect[0][0]+window_xmin), round(minRect[0][1]+window_ymin)), 
                    (round(minRect[1][0]), round(minRect[1][1])), 
                     round(minRect[2], 2))
        rotbox_ = cv2.boxPoints(minRect_)     # minRect to boxVertexes: array([[x0,y0], [x1,y1], [x2,y2], [x3,y3]], dtype=float32)
        rotbox = np.int32(rotbox_).tolist()   # array to list [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    else:
        raise Exception("unknown method")
    return rotbox


def get_orientation(contour):  
    """
    Purpose: get orientation by PCA and draw the axises of components
    Args:
        contour: a 3-d array of contour, [[[x0,y0]], [[x1,y1]], ... , [[xn,yn]]].shape() = (npoints, 1, 2)
    Returns:
        angle: the angle of the principle component orientation (in radians)
    """  
    # optional - subsample contour, reduce the amount of calculation of pca 
    contour = contour_subsample(contour, subsampleRate=None, subsampleNum=20) 
    # transform to PCA input data  
    data_pts = np.empty(shape=(len(contour), 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = contour[i,0,0]
        data_pts[i,1] = contour[i,0,1]
    # Perform PCA 
    mean = np.empty((0))
    mean, eigenVectors, eigenValues = cv2.PCACompute2(data_pts, mean)
    # calculate center and angle
    center_point = (int(mean[0,0]), int(mean[0,1]))
    angle = atan2(eigenVectors[0,1], eigenVectors[0,0]) - pi/2 # 0度角方向垂直向下，正方向为顺时针
    
    return center_point, angle, eigenVectors, eigenValues


def rotate_point(point, center, angle):
    """
    Purpose: rotate `point` by `angle`, centered by `center`
    Args:
        point: the point to be rotated, a tuple of coordinate 
        center: the center point: a tuple of coordinate 
        angle: the rotation angle in radian
    Returns:
        new_point: the point after rotating
    """
    # unpack tuple
    pt_x, pt_y = list(point)
    cntr_x, cntr_y = list(center)
    # rotation maxtrix
    npt_x = (pt_x - cntr_x)*cos(-angle) - (pt_y - cntr_y)*sin(-angle) + cntr_x
    npt_y = (pt_x - cntr_x)*sin(-angle) + (pt_y - cntr_y)*cos(-angle) + cntr_y
    new_point = (npt_x, npt_y)
    return new_point


def rotate_contour(contour, center, angle):
    """
    Purpose: rotate contour `angle`, centered by `center`, point by point
    Args:
        contour: the contour to be rotated, [ [[x0,y0]], [[x1,y1]], ... ].shape() = (npoints, 1, 2)
        center: the center point: a tuple of coordinate 
        angle: the rotation angle in radian
    Returns:
        rotated_contour: the contour after rotating
    """
    rotated_contour = contour.copy() # 拷贝副本, 不影响原本
    # 轮廓点逐个旋转
    for i, ctr_pt in enumerate(contour):
        rotated_contour[i,0,0], rotated_contour[i,0,1] = rotate_point(tuple(ctr_pt[0]), center, angle)
    return rotated_contour


def draw_rotbox(img, rotbox, color, thickness=1, rotbox_mode=OBB_MODE.PCA):
    """
    Purpose: deaw rotation box in the image
    Args:
        img: the canvas to be drawn the rotbox 
        rotbox: rotation box, list: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        color: the color of the rotation box
        thickness: the thickness of box line
        rotbox_mode: it should be same with the method of get_rotbox(), OBB_MODE.PCA or OBB_MODE.CONVEXHULL
    """
    if len(img.shape) == 2:
        img = cv2.merge([img, img, img])
    if rotbox_mode is OBB_MODE.PCA:
        for i in range(len(rotbox)):
            start_point = (int(rotbox[i][0]), int(rotbox[i][1]))
            end_point = (int(rotbox[(i+1)%4][0]), int(rotbox[(i+1)%4][1]))
            cv2.line(img, start_point, end_point, color, thickness)
        cv2.waitKey(0)
    elif rotbox_mode is OBB_MODE.CONVEXHULL:
        rotbox_points = np.array(rotbox, dtype=np.int32)    # list to array
        cv2.drawContours(img, [rotbox_points], 0, color, thickness)
        cv2.imshow('rotbox', img)
        cv2.waitKey(0)
    else:
        raise Exception("unknown rotbox_mode")