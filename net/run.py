import argparse
from fallDownDetectYolo import FallDownDetectYolo
import sys

if __name__ == "__main__":
    # 初始化参数解析器
    parser = argparse.ArgumentParser(
        description="基于 YOLOv5 的跌倒检测器，支持图片和视频输入，支持 PyTorch 和 ONNX 格式模型挂载",
        epilog="命令行输入示例: python run.py --yolov5_path yolov5 --weight_path yolov5/yolov5_best.pt --image_path image.png --video_path video.avi --model_type pt --input_type image",
    )

    # 添加参数
    parser.add_argument("--yolov5_path", type=str, required=True, help="YOLOv5 本地路径")
    parser.add_argument("--weight_path", type=str, required=True, help="模型权重路径 (.pt 或 .onnx)")
    parser.add_argument("--image_path", type=str, help="待检测的图片路径")
    parser.add_argument("--video_path", type=str, help="待检测的视频路径")
    parser.add_argument("--model_type", type=str, required=True, choices=['pt', 'onnx'], help="模型类型：pt 或 onnx")
    parser.add_argument("--input_type", type=str, required=True, choices=['image', 'video'], help="输入类型：image 或 video")
    parser.add_argument("--save", action='store_true', help="是否保存检测结果")
    parser.add_argument("--save_path", type=str, default=None, help="检测结果保存路径（图片或视频）")
    parser.add_argument("--fps", type=int, default=30, help="保存视频的帧率（仅在视频模式下有效）")

    # 解析参数
    args = parser.parse_args()

    # 参数验证
    if args.input_type == "image" and not args.image_path:
        print("错误: 在图像模式下，必须指定 --image_path")
        sys.exit(1)
    if args.input_type == "video" and not args.video_path:
        print("未指定 video_path ，将使用摄像头捕获视频流")
        args.video_path = None
        
    # 创建检测器
    detector = FallDownDetectYolo(
        yolov5_path=args.yolov5_path,
        weight_path=args.weight_path,
        image_path=args.image_path if args.input_type == 'image' else '',
        video_path=args.video_path if args.input_type == 'video' else None,
        pt_or_onnx=args.model_type
    )

    # 运行检测器
    if args.input_type == "image":
        detector.img_inference(save=args.save, save_path=args.save_path)
    elif args.input_type == "video":
        detector.video_inference(save=args.save,
                                 save_path=args.save_path if args.save_path else "output.mp4",
                                 fps=args.fps)