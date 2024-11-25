import sys                  # 导入 sys 模块, 用于退出程序
import torch                # 导入 torch 模块, 用于加载模型
import numpy as np          # 导入 numpy 模块, 用于处理图像
import pandas as pd         # 导入 pandas 模块, 用于处理检测结果
import cv2                  # 导入 OpenCV 模块, 用于读取图像
import platform             # 导入 platform 模块, 用于判断操作系统
import pathlib              # 导入 pathlib 模块, 用于处理路径
import onnxruntime as ort   # 导入 onnxruntime 模块, 用于加载 ONNX 模型

# Letterbox 填充
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class FallDownDetectYolo:
    def __init__(self, yolov5_path: str, weight_path: str, image_path: str, pt_or_onnx: str):
        # 检测操作系统类型保证模型路径的兼容性
        if platform.system() != 'Windows':
            pathlib.WindowsPath = pathlib.PosixPath

        # 初始化模型、图片路径与运行模式
        self.yolov5_path = yolov5_path
        self.weight_path = weight_path
        self.image_path = image_path
        self.pt_or_onnx = pt_or_onnx

        # Load class names
        self.class_names = self.load_class_names()
        print("\n类别映射:")
        for class_id, class_name in enumerate(self.class_names):
            print(f"Class ID: {class_id}, Class Name: {class_name}")

        # 加载模型
        if self.pt_or_onnx == 'onnx':
            self.session = ort.InferenceSession(str(self.weight_path))
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print("已加载 ONNX 模型")

        elif self.pt_or_onnx == 'pt':
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {self.device}")
            self.model = torch.hub.load(
                str(self.yolov5_path),       # YOLOv5 本地路径
                'custom',                    # 自定义模型名称
                path=str(self.weight_path),  # 自定义模型权重路径
                source='local'               # 指定从本地加载
            )
            self.model.to(self.device)
            self.model.eval()
            print("已加载 PyTorch 模型")

        else:
            print("模型类型错误")
            sys.exit(1)

    # 加载类别名称
    def load_class_names(self):
        class_names = ['normal', 'down']
        return class_names

    # 将 [x_center, y_center, width, height] 转换为 [x_min, y_min, x_max, y_max]
    def xywh_to_xyxy(self, boxes):
        """
        将 [x_center, y_center, width, height] 转换为 [x_min, y_min, x_max, y_max]
        """
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return np.stack([x_min, y_min, x_max, y_max], axis=1)

    # 推理函数（基于图片输入）
    def img_inference(self):
        # pt 模型推理
        if self.pt_or_onnx == 'pt':
            model = self.model

            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device == 'cuda':
                print('正在使用 Nvidia GPU 进行推理')
            else:
                print('正在使用 CPU 进行推理')
            model.to(device)

            # 读取图像与预处理
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 检查图像是否成功读取
            if img is None:
                print(f"无法读取图片文件: {self.image_path}")
                sys.exit(1)

            # 进行推理
            results = model(img)

            # 打印检测结果
            results.print()

            # 将检测结果转换为 Pandas DataFrame
            df = results.pandas().xyxy[0]  # 获取第一张图像的结果

            # 添加类别名称
            df['class_name'] = df['class'].apply(lambda x: model.names[int(x)])

            # 打印感兴趣的列
            print(df[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

            # 显示检测结果图像
            results.show()

        # onnx 模型推理
        elif self.pt_or_onnx == 'onnx':
            preprocessed_img, original_img, ratio, padding = self.preprocess_image()
            img = preprocessed_img
            outputs = self.session.run(self.output_names, {self.input_name: img})[0]
            df = self.postprocess_image(outputs, ratio, padding)
            detected_img = self.draw_detections(original_img, df)
            print("\nONNX 推理结果:")
            print(df[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
            cv2.imshow('ONNX Detections', detected_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def preprocess_image(self, input_size=640):
        # 读取原始图像
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片文件: {self.image_path}")

        # 调用 letterbox 函数保持宽高比缩放，并填充到目标尺寸
        img_resized, ratio, (dw, dh) = letterbox(img, new_shape=(input_size, input_size))

        # 转换颜色空间 (BGR -> RGB)
        img_rgb = img_resized[:, :, ::-1]  # BGR to RGB

        # 归一化
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # 转换维度顺序 HWC -> CHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # 添加批量维度 NCHW
        img_expanded = np.expand_dims(img_transposed, axis=0)

        # 返回预处理后的图像、原始图像，以及缩放和填充信息
        return img_expanded, img, ratio, (dw, dh)

    def postprocess_image(self, outputs, ratio, padding, conf_threshold=0.5, iou_threshold=0.5):
        outputs = np.squeeze(outputs, axis=0)  # (1, 25200, 7) -> (25200, 7)

        # 提取信息 (假设输出为 [x_center, y_center, width, height, confidence, class])
        boxes_xywh = outputs[:, :4]  # [x_center, y_center, width, height]
        confidences = outputs[:, 4]  # 置信度
        class_scores = outputs[:, 5:]  # 每个类别的概率

        # 获取最高类别及其置信度
        class_ids = np.argmax(class_scores, axis=1)
        confidences = confidences * class_scores[np.arange(len(class_scores)), class_ids]

        # 转换坐标为 [x_min, y_min, x_max, y_max]
        boxes_xyxy = self.xywh_to_xyxy(boxes_xywh)

        # 应用 NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confidences.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            boxes_xyxy = boxes_xyxy[indices]
            confidences = confidences[indices]
            class_ids = class_ids[indices]
        else:
            return pd.DataFrame(columns=['class_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])

        # 坐标还原
        boxes_xyxy[:, [0, 2]] -= padding[0]  # 减去横向填充
        boxes_xyxy[:, [1, 3]] -= padding[1]  # 减去纵向填充
        boxes_xyxy[:, [0, 2]] /= ratio[0]  # 恢复宽度比例
        boxes_xyxy[:, [1, 3]] /= ratio[1]  # 恢复高度比例

        # 创建 DataFrame
        df_onnx = pd.DataFrame({
            'class_id': class_ids,
            'confidence': confidences,
            'xmin': boxes_xyxy[:, 0],
            'ymin': boxes_xyxy[:, 1],
            'xmax': boxes_xyxy[:, 2],
            'ymax': boxes_xyxy[:, 3]
        })
        df_onnx['class_name'] = df_onnx['class_id'].apply(lambda x: self.class_names[x] if x < len(self.class_names) else str(x))

        return df_onnx

    def draw_detections(self, img, df_onnx):
        for _, row in df_onnx.iterrows():
            x_min = int(row['xmin'])
            y_min = int(row['ymin'])
            x_max = int(row['xmax'])
            y_max = int(row['ymax'])

            # 根据类别名称选择颜色
            if row['class_name'] == 'down':  # 根据类别名称判断
                color = (0, 0, 255)  # 红色框
            else:
                color = (0, 255, 0)  # 绿色框

            # 绘制边界框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # 绘制标签
            label = f"{row['class_name']} {row['confidence']:.2f}"
            cv2.putText(img, label, (x_min, y_min + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return img

    # 推理函数（基于视频流输入）
    def video_inference(self, source=0, display=True, save=False, save_path='output.avi'):
        """
        基于视频流（摄像头或视频文件）进行推理。

        :param source: 视频源，0 表示默认摄像头，或视频文件路径
        :param display: 是否显示推理结果
        :param save: 是否保存推理结果视频
        :param save_path: 保存视频的路径
        """
        # 打开视频源
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频源: {source}")
            sys.exit(1)

        # 获取视频信息
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # 默认帧率

        # 设置视频写入
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

        print("开始视频推理，按 'q' 键退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或无法读取帧")
                break

            # 进行推理
            if self.pt_or_onnx == 'pt':
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(img_rgb)

                # 获取检测结果
                detections = results.pandas().xyxy[0].to_dict(orient='records')  # 列表字典

                # 转换为自定义格式
                formatted_detections = []
                for det in detections:
                    formatted_detections.append({
                        'box': [int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])],
                        'confidence': float(det['confidence']),
                        'class_id': int(det['class'])  # 假设类别 ID 为 0 或 1
                    })

                # 绘制检测框
                annotated_frame = self.draw_detections(frame.copy(), formatted_detections)

            elif self.pt_or_onnx == 'onnx':
                # 预处理图像
                img_resized = cv2.resize(frame, (640, 640))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
                img_expanded = np.expand_dims(img_transposed, axis=0)   # 添加批量维度

                # 进行推理
                outputs = self.session.run(self.output_names, {self.input_name: img_expanded})[0]

                # 后处理
                detections = self.postprocess_onnx(outputs, conf_threshold=0.5, iou_threshold=0.45)

                # 绘制检测框
                annotated_frame = self.draw_detections(img_resized, detections)

            # 显示结果
            if display:
                cv2.imshow('Video Inference', annotated_frame)

            # 保存结果
            if save:
                out.write(annotated_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("手动终止推理")
                break

        # 释放资源
        cap.release()
        if save:
            out.release()
        cv2.destroyAllWindows()