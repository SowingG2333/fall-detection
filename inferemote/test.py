from inferemote.testing import AiremoteTest
import cv2 as cv
from fall import FallDownDetectYolo

def make_image(img, detections):
    """
    根据检测结果绘制边界框和标签
    """
    for detection in detections:
        x_min = round(detection["xmin"])
        y_min = round(detection["ymin"])
        x_max = round(detection["xmax"])
        y_max = round(detection["ymax"])
        confidence = detection["confidence"]
        class_name = detection["class_name"]

        # 根据类别名称选择颜色
        if class_name == "down":    # 根据类别名称判断
            color = (0, 0, 255)     # 红色框
        elif class_name == "normal":
            color = (0, 255, 0)     # 绿色框

        # 绘制边界框
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # 绘制标签
        label = f"{class_name} {confidence:.2f}"
        cv.putText(img, label, (x_min, y_min + 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return img

class MyTest(AiremoteTest, FallDownDetectYolo):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ''' An airemote object '''
        self.air = FallDownDetectYolo()

    def run(self, image):
        bboxes = self.air.inference(image)
        new_image = make_image(image, bboxes)
        return new_image

if __name__ == '__main__':
    test_img = '/Users/sowingg/ict-class/homework/task5/image.png'

    air = FallDownDetectYolo()
    t = MyTest(remote='adk')
    t.start(input=test_img, mode='show')