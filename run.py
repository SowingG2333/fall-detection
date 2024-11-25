from fallDownDetectYolo import FallDownDetectYolo

if __name__ == "__main__":
    # 示例参数
    yolov5_path = 'yolov5'  # YOLOv5 本地路径
    weight_path_pt = 'yolov5/yolov5_best.pt'     # PyTorch 模型路径
    weight_path_onnx = 'yolov5/yolov5_best.onnx' # ONNX 模型路径
    image_path = 'image.png'  # 图片路径

    # 选择模型类型：'pt' 或 'onnx'
    model_type = 'onnx'

    # 初始化检测器
    detector = FallDownDetectYolo(
        yolov5_path=yolov5_path,
        weight_path=weight_path_pt if model_type == 'pt' else weight_path_onnx,
        image_path=image_path,
        pt_or_onnx=model_type
    )

    # 视频源：0 表示默认摄像头，或替换为视频文件路径
    video_source = 0  # 使用摄像头
    # video_source = 'path/to/video.mp4'  # 使用视频文件

    # 调用视频推理函数
    detector.img_inference()