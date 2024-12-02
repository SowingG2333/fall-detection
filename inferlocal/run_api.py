from fallDownDetectYolo import FallDownDetectYolo

# 初始化检测器
detector = FallDownDetectYolo(
    yolov5_path='yolov5',
    weight_path='yolov5/model.onnx',
    image_path='image.png',
    video_path=None,
    pt_or_onnx='onnx'
)

# 对单张图片进行推理
detector.img_inference()