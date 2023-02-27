import sys
import os
import cv2
import numpy as np
import paddle
import math
import time
import collections
import copy
import tarfile
import urllib.request

from PIL import Image
from pathlib import Path

from openvino.runtime import Core
from . import pre_post_processing as processing

package_directory = Path(__file__).parent

def run_model_download(model_url, model_file_path):
    """
    Download pre-trained models from PaddleOCR resources

    Parameters:
        model_url: url link to pre-trained models
        model_file_path: file path to store the downloaded model, relative to the package directory
    """
    model_name = model_url.split("/")[-1]

    if (package_directory / model_file_path).is_file():
        pass
    else:
        # Download the model from the server, and untar it.
        print("Downloading the pre-trained model... May take a while...")

        # Create a directory.
        (package_directory / "model").mkdir(exist_ok=True)
        urllib.request.urlretrieve(model_url, str(package_directory / "model" / model_name))
        print("Model Downloaded")

        file = tarfile.open(str(package_directory / "model" / model_name))
        res = file.extractall(str(package_directory / "model"))
        file.close()
        if not res:
            print(f"Model Extracted to {model_file_path}.")
        else:
            print("Error Extracting the model. Please check the network.")
            
#-------------------------- Download Models ---------------------------#
det_model_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar"
det_model_file_path = package_directory / "model/ch_PP-OCRv3_det_infer/inference.pdmodel"
run_model_download(det_model_url, det_model_file_path)

rec_model_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar"
rec_model_file_path = package_directory / "model/ch_PP-OCRv3_rec_infer/inference.pdmodel"
run_model_download(rec_model_url, rec_model_file_path)

#--------------------------- Text Detection ---------------------------#
# Initialize OpenVINO Runtime for text detection.
core = Core()
det_model_file_path = package_directory / "model/ch_PP-OCRv3_det_infer/inference.pdmodel"
det_model = core.read_model(model=det_model_file_path)
det_compiled_model = core.compile_model(model=det_model, device_name="CPU")

# Get input and output nodes for text detection.
det_input_layer = det_compiled_model.input(0)
det_output_layer = det_compiled_model.output(0)

#--------------------------- Text Recognition ---------------------------#
# Read the model and corresponding weights from a file.
rec_model_file_path = package_directory / "model/ch_PP-OCRv3_rec_infer/inference.pdmodel"
rec_model = core.read_model(model=rec_model_file_path)

# Assign dynamic shapes to every input layer on the last dimension.
for input_layer in rec_model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = -1
    rec_model.reshape({input_layer: input_shape})

rec_compiled_model = core.compile_model(model=rec_model, device_name="CPU")

# Get input and output nodes.
rec_input_layer = rec_compiled_model.input(0)
rec_output_layer = rec_compiled_model.output(0)

# Preprocess for text detection.
def image_preprocess(input_image, size):
    """
    Preprocess input image for text detection

    Parameters:
        input_image: input image 
        size: value for the image to be resized for text detection model
    """
    img = cv2.resize(input_image, (size, size))
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    # NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)

def post_processing_detection(frame, det_results):
    """
    Postprocess the results from text detection into bounding boxes

    Parameters:
        frame: input image 
        det_results: inference results from text detection model
    """   
    ori_im = frame.copy()
    data = {'image': frame}
    data_resize = processing.DetResizeForTest(data)
    data_list = []
    keep_keys = ['image', 'shape']
    for key in keep_keys:
        data_list.append(data_resize[key])
    img, shape_list = data_list

    shape_list = np.expand_dims(shape_list, axis=0) 
    pred = det_results[0]    
    if isinstance(pred, paddle.Tensor):
        pred = pred.numpy()
    segmentation = pred > 0.3

    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        mask = segmentation[batch_index]
        boxes, scores = processing.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
        boxes_batch.append({'points': boxes})
    post_result = boxes_batch
    dt_boxes = post_result[0]['points']
    dt_boxes = processing.filter_tag_det_res(dt_boxes, ori_im.shape)    
    return dt_boxes

def prep_for_rec(dt_boxes, frame):
    """
    Preprocessing of the detected bounding boxes for text recognition

    Parameters:
        dt_boxes: detected bounding boxes from text detection 
        frame: original input frame 
    """
    ori_im = frame.copy()
    img_crop_list = [] 
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = processing.get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
        
    img_num = len(img_crop_list)
    # Calculate the aspect ratio of all text bars.
    width_list = []
    for img in img_crop_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    
    # Sorting can speed up the recognition process.
    indices = np.argsort(np.array(width_list))
    return img_crop_list, img_num, indices

def batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num):
    """
    Batch for text recognition

    Parameters:
        img_crop_list: processed detected bounding box images 
        img_num: number of bounding boxes from text detection
        indices: sorting for bounding boxes to speed up text recognition
        beg_img_no: the beginning number of bounding boxes for each batch of text recognition inference
        batch_num: number of images for each batch
    """
    norm_img_batch = []
    max_wh_ratio = 0
    end_img_no = min(img_num, beg_img_no + batch_num)
    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    return norm_img_batch

# Preprocess for text recognition.
def resize_norm_img(img, max_wh_ratio):
    """
    Resize input image for text recognition

    Parameters:
        img: bounding box image from text detection 
        max_wh_ratio: value for the resizing for text recognition model
    """
    rec_image_shape = [3, 48, 320]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    character_type = "ch"
    if character_type == "ch":
        imgW = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def run_paddle_ocr(img):
    """
    Main function for text recognition.

    Args:
        img (_type_): Image to be processed.
    """
    # Rescale image to 640x640
    scale = 1280 / max(img.shape)
    if scale < 1:
        img = cv2.resize(src=img, dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_CUBIC)
    test_image = image_preprocess(img, 160)

    # Perform the inference step.
    det_results = det_compiled_model([test_image])[det_output_layer]
    dt_boxes = post_processing_detection(img, det_results)
    
    
    # Preprocess detection results for recognition
    dt_boxes = processing.sorted_boxes(dt_boxes)  
    batch_num = 6
    img_crop_list, img_num, indices = prep_for_rec(dt_boxes, img)
    
    # For storing recognition results, include two parts:
    # txts are the recognized text results, scores are the recognition confidence level. 
    rec_res = [['', 0.0]] * img_num
    txts = [] 
    scores = []

    for beg_img_no in range(0, img_num, batch_num):
        # Recognition starts from here.
        norm_img_batch = batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num)

        # Run inference for text recognition. 
        rec_results = rec_compiled_model([norm_img_batch])[rec_output_layer]

        # Postprocessing recognition results.
        postprocess_op = processing.build_post_process(processing.postprocess_params)
        rec_result = postprocess_op(rec_results)
        for rno in range(len(rec_result)):
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]   
        if rec_res:
            txts = [rec_res[i][0] for i in range(len(rec_res))] 
            scores = [rec_res[i][1] for i in range(len(rec_res))]

    results = txts
    
    return results
