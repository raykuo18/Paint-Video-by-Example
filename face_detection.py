from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# from modelscope.utils.cv.image_utils import draw_face_detection_no_lm_result
from modelscope.preprocessors.image import LoadImage
import cv2
import numpy as np

def draw_face_detection_no_lm_result(img_path, detection_result):
    bboxes = np.array(detection_result['boxes'])
    scores = np.array(detection_result['scores'])
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read img: {img_path}"
    for i in range(len(scores)):
        bbox = bboxes[i].astype(np.int32)
        x1, y1, x2, y2 = bbox
        score = scores[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            f'{score:.2f}', (x1, y2),
            1,
            1.0, (0, 255, 0),
            thickness=1,
            lineType=8)
    print(f'Found {len(scores)} faces')
    return img

def main():
    mog_face_detection_func = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')
    src_img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/mog_face_detection.jpg'
    raw_result = mog_face_detection_func(src_img_path)
    print('face detection output: {}.'.format(raw_result))

    # load image from url as rgb order
    src_img = LoadImage.convert_to_ndarray(src_img_path)
    # save src image as bgr order to local
    src_img  = cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite('src_img.jpg', src_img) 
    # draw dst image from local src image as bgr order
    dst_img = draw_face_detection_no_lm_result('src_img.jpg', raw_result)
    # save dst image as bgr order to local
    cv2.imwrite('dst_img.jpg', dst_img)
    # # show dst image by rgb order
    # import matplotlib.pyplot as plt
    # dst_img  = cv2.cvtColor(np.asarray(dst_img), cv2.COLOR_BGR2RGB)
    # plt.imshow(dst_img)
    
def create_mask(img_path, detection_result):
    bboxes = np.array(detection_result['boxes'])
    scores = np.array(detection_result['scores'])
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read img: {img_path}"
    img = np.zeros(img.shape, dtype = "uint8")
    for i in range(len(scores)):
        bbox = bboxes[i].astype(np.int32)
        x1, y1, x2, y2 = bbox
        score = scores[i]
        expand = 10
        cv2.rectangle(img, (x1-expand, y1-expand), (x2+expand, y2+expand), (255, 255, 255), -1)
        # cv2.putText(
        #     img,
        #     f'{score:.2f}', (x1, y2),
        #     1,
        #     1.0, (0, 255, 0),
        #     thickness=1,
        #     lineType=8)
    print(f'Found {len(scores)} faces')
    return img
    
def face_detect(src_img_path):
    mog_face_detection_func = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')
    raw_result = mog_face_detection_func(src_img_path)
    mask = create_mask(src_img_path, raw_result)
    mask = cv2.resize(mask, (512, 512))
    cv2.imwrite('mask.jpg', mask)
    
    # Resize source image
    img = cv2.imread(src_img_path)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite('img.jpg', img)
    
if __name__ == '__main__':
    # main()
    # face_detect('data/2003/01/02/big/img_7.jpg')
    face_detect('img_10_.jpg')
    