import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
import numpy as np
import imageio
import skimage.io
from io import BytesIO
import matplotlib.pyplot as plt
import PIL

if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                        r_model_path="./original_model/rnet_epoch.pt",
                                        o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # print(type(ret), type(frame))
        img_bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        save_name = str(np.random.uniform()) + 'out_put.jpg'
        mark_face_frame = vis_face(img_bg, bboxs, landmarks, save_name, save=False, show=False)
        buffer_ = BytesIO()  # using buffer,great way!
        # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
        mark_face_frame.savefig(buffer_, format='png')
        buffer_.seek(0)
        # 用PIL或CV2从内存中读取
        dataPIL = PIL.Image.open(buffer_)
        # 转换为nparrary，PIL转换就非常快了,data即为所需
        data = np.asarray(dataPIL)
        cv2.imshow("capture", data[:, :, (2, 1, 0)])
        buffer_.close()
        # mark_face_frame.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
