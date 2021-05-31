from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
import argparse
import progressbar

from convert_voc_result import resolve_result, store_voc_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection inference')
    parser.add_argument('--cfg', 
                        dest='config',
                        help='inference config file path',
                        type=str, 
                        default=None
                        )
    parser.add_argument('--checkpoint', 
                        dest='checkpoint',
                        help='checkpoint file path',
                        type=str, 
                        default=None
                        )
    parser.add_argument('--in', 
                        dest='img_input',
                        help='the input path of image to inference',
                        type=str, 
                        default=None
                        )
    parser.add_argument('--out', 
                        dest='img_output',
                        help='the output path of image that has inferenced',
                        type=str, 
                        default=None
                        )

    args = parser.parse_args()
    return args
 

def main():
    # 默认路径
    # CONFIG_FILE = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py' # 模型的配置文件
    # CONFIG_FILE = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' # 模型的配置文件
    # CONFIG_FILE = 'configs/yolo/yolov3_d53_mstrain-608_273e_coco.py' # 模型的配置文件
    CONFIG_FILE = 'configs/ssd/ssd300_coco.py' # 模型的配置文件
    # CONFIG_FILE = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py' # 模型的配置文件
    CHECKPOINT_FILE = 'work_dirs/ssd300_coco/epoch_12.pth' # 训练好的模型权重
    # IMG_PATH = '/home/watson/Documents/mask_THzDatasets/val'
    IMG_PATH = '/home/watson/Documents/mask_THzDatasets/test2/imgs'
    RESULT_ROOT = "/home/watson/Documents/mmdet_result/{}".format(os.path.splitext(os.path.basename(CONFIG_FILE))[0])

    # 解析参数
    args = parse_args()
    if args.config is not None:
        CONFIG_FILE = args.config
    if args.checkpoint is not None:
        CHECKPOINT_FILE = args.checkpoint
    if args.img_input is not None:
        IMG_PATH = args.img_input
    if args.img_output is not None:
        RESULT_ROOT = args.img_output

    RESULT_IMG_PATH = os.path.join(RESULT_ROOT, "imgs")
    RESULT_XML_PATH = os.path.join(RESULT_ROOT, "xmls")
    if not os.path.exists(RESULT_IMG_PATH):
        os.makedirs(RESULT_IMG_PATH)
    if not os.path.exists(RESULT_XML_PATH):
        os.makedirs(RESULT_XML_PATH)

    # 初始化模型
    print('=== initialing detector ===')
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE)
    
    # 推断图片
    print('inference start!')
    img_list = os.listdir(IMG_PATH)
    img_cnt = 0
    img_totalNum = len(img_list)
    bar = progressbar.ProgressBar(max_value=img_totalNum).start()
    for img_fileName in img_list:
        img_cnt += 1
        bar.update(img_cnt)
        img_fullFileName = os.path.join(IMG_PATH, img_fileName)
        result_img_fullName = os.path.join(RESULT_IMG_PATH, img_fileName)
        result_xml_fullName = os.path.join(RESULT_XML_PATH, img_fileName.split('.')[0] + "_result.xml")

        result = inference_detector(model, img_fullFileName)
        img = model.show_result(img_fullFileName, result, score_thr=0.3, show=False, out_file=result_img_fullName)
        # show_result_pyplot(model, img_fullFileName, result, score_thr=0.3)

        detections = resolve_result(result, class_names=model.CLASSES, score_thr=0.3)
        
        store_voc_result(img_fullFileName, detections, result_xml_fullName)
        
    bar.finish()
    print('inference finished!')


if __name__ == '__main__':
    main()