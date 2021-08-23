import cv2 
import numpy as np
import os 
os.chdir('mmocr')
from mmocr.utils.ocr import MMOCR
os.chdir('..')
from pan.predict import Pytorch_model

from textClassify.product_classifier_infer import ClassifierInfer


def crop_text(polys, img):
    crop_imgs = []
    for poly in polys:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        crop_imgs.append(img[y:y+h, x:x+w, :].copy())
    return crop_imgs

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# get 4 points of rectangle from poly
def get_box_from_poly(pts):
    pts = pts.reshape((-1,2)).astype(int)
    x,y,w,h = cv2.boundingRect(pts)
    return np.array([x,y,x+w,y+h])

def ensemble(pts1, pts2):
    # take pts2 first then pts1
    iou_thres = 0.5
    pts = []
    box2 = []
    for poly in pts2:
        box2.append(get_box_from_poly(poly))
    
    rm = []
    for count, poly in enumerate(pts1):
        box = get_box_from_poly(poly)
        iou_score = np.array([bb_intersection_over_union(box, box2i) for box2i in box2])
        if np.any(iou_score) > iou_thres:
            rm.append(count)
    pts1 = np.delete(pts1, rm, axis=0)
    
    if pts1.size == 0:
        pts = pts2
        if pts2.size == 0:
            pts = []
    else:
        pts = np.vstack((pts1, pts2))   
    return pts


def convert_boxPoint2Yolo(pts, imgShape):
    yolo = []
    height = imgShape[0]
    width = imgShape[1]
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        x = (x+w/2)/width
        y = (y+h/2)/height
        yolo.append(np.array([x,y,w/width, h/height]))
    return yolo

def sort_pts(pts, max_pts):
    heights = []    # sort by height
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        heights.append(h)
    new_poly = pts[np.argsort(heights)][::-1]
    return new_poly[:max_pts]

def textSpotting(detect1, detect2, recog, img, max_word=16):
    mmocr_det_res = detect2.readtext(img=[img.copy()])
    pts_mmocr = np.array([np.array(pts[:8]).reshape((-1,2)) for pts in mmocr_det_res[0]['boundary_result']])

    preds, pts_pan, t = detect1.predict(img=img.copy())
    pts_pan[np.where(pts_pan < 0)] = 0

    pts = ensemble(pts_mmocr, pts_pan) 
    # sort and take up to max_word poly
    pts = sort_pts(pts, max_word)
    yolo = convert_boxPoint2Yolo(pts, img.shape)
    
    crop_imgs = crop_text(pts, img)
    
    result_recog = recog.readtext(img=crop_imgs.copy(), batch_mode=True, single_batch_size=max_word)
    
    temp = {'boxPoint': None, 'boxYolo': None, 'text': None, 'text_score': None}
    result = []
    for count, poly in enumerate(pts):
        temp1 = temp.copy()
        temp1['boxPoint'] = poly
        temp1['boxYolo'] = yolo[count]
        temp1['text'] = result_recog[count]['text']
        temp1['text_score'] = result_recog[count]['score']
        result.append(temp1)
        
    return result

def textClassify(model1, model2, model3, pre_result):
    text = ' '.join([res['text'] for res in pre_result])
    level1 = model1.product_predict([text])
    level2 = model2.product_predict([text])
    level3 = model3.product_predict([text])
    has_age = model1.check_has_age(text)
    
    pre_result.append((level1, level2, level3, has_age))
    return pre_result
    


if __name__ == "__main__":
    # Load models into memory
    mmocr_detect = MMOCR(det='DB_r50', recog=None)
    mmocr_recog = MMOCR(det=None, recog='SAR')

    model_path = 'pan/pretrain/pannet_wordlevel.pth'
    pan_detect = Pytorch_model(model_path, gpu_id=0)

    classifyModel_level1 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level1_f192.15.pkl')
    classifyModel_level2 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level2_f19306.pkl')
    classifyModel_level3 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level3_f1_8117.pkl')
    
    # inference
    img = cv2.imread('TextSpottingAndTextClassify/img_1.jpg')
    
    result = textSpotting(pan_detect, mmocr_detect, mmocr_recog, img)

    result = textClassify(classifyModel_level1, classifyModel_level2, classifyModel_level3, result)
    
    #save txt
    
    
    #save image
    
    