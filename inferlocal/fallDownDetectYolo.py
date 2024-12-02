import sys                  # 导入 sys 模块, 用于退出程序
import torch                # 导入 torch 模块, 用于加载模型
import numpy as np          # 导入 numpy 模块, 用于处理图像
import pandas as pd         # 导入 pandas 模块, 用于处理检测结果
import cv2                  # 导入 OpenCV 模块, 用于读取图像
import platform             # 导入 platform 模块, 用于判断操作系统
import pathlib              # 导入 pathlib 模块, 用于处理路径
import onnxruntime as ort   # 导入 onnxruntime 模块, 用于加载 ONNX 模型

# Letterbox 填充（yolov5特有的填充方式）
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
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
    def __init__(self, yolov5_path: str, weight_path: str, image_path: str, pt_or_onnx: str, video_path=0):
        '''
        初始化检测器，包括 yolov5 路径，模型路径，图片路径和模型类型
        '''
        # 检测操作系统类型保证模型路径的兼容性
        if platform.system() != 'Windows':
            pathlib.WindowsPath = pathlib.PosixPath

        # 初始化模型、图片路径与运行模式
        self.yolov5_path = yolov5_path
        self.weight_path = weight_path
        self.image_path = image_path
        self.video_source = video_path
        self.pt_or_onnx = pt_or_onnx

        # 加载类别名称
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

    def load_class_names(self):
        '''
        加载类别名称，0对应 normal ，1对应 down
        '''
        class_names = ['normal', 'down']
        return class_names

    def xywh_to_xyxy(self, boxes):
        """
        yolo 模型的原始输出为 [x_center, y_center, width, height] 格式的坐标
        如果不使用官方提供的API（官方API仅支持pt格式加载），需要手动转换坐标格式为 [x_min, y_min, x_max, y_max]以进行框图绘制
        本函数实现从 [x_center, y_center, width, height] 到 [x_min, y_min, x_max, y_max] 的转换
        """
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        return np.stack([x_min, y_min, x_max, y_max], axis=1)

    def preprocess_image(self, img, input_size=640):
        '''
        图像预处理，使用 letterbox 函数保持宽高比缩放，并填充到目标尺寸（640x640）
        同时进行颜色空间转换，归一化，维度转换和批量维度添加的操作
        '''
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

        # 返回预处理后的图像、缩放和填充信息
        return img_expanded, ratio, (dw, dh)

    def postprocess_image(self, outputs, ratio, padding, conf_threshold=0.6, iou_threshold=0.3):
        '''
        图像后处理，首先将图像输出进行切片，提取坐标，置信度和类别概率
        由于输出带有批次维度，因此首先需要进行压缩，然后获取目标类别和置信度
        接着用 NMS 算法进行框的筛选，最后将坐标还原到原始图像尺寸
        '''
        outputs = np.squeeze(outputs, axis=0)  # (1, num_predictions, 7) -> (num_predictions, 7)
        print(outputs)
        # 提取信息 (输出为 [x_center, y_center, width, height, confidence, class])
        boxes_xywh = outputs[:, :4]  # [x_center, y_center, width, height]
        confidences = outputs[:, 4]  # 置信度
        class_scores = outputs[:, 5:]  # 每个类别的概率

        # 获取最高类别及其置信度
        class_ids = np.argmax(class_scores, axis=1)
        confidences = confidences * class_scores[np.arange(len(class_scores)), class_ids]

        # 转换坐标为 [x_min, y_min, x_max, y_max]
        boxes_xyxy = self.xywh_to_xyxy(boxes_xywh)

        # 应用 NMS（非极大值抑制，选出置信度最高的框图，与候选框计算交并比并对重叠框进行删除）
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confidences.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )

        print(indices)
        print(len(indices))

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
        '''
        根据检测结果绘制边界框和标签
        '''
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

    # 推理函数（基于图片输入）
    def img_inference(self, save=False, save_path='output.jpg'):
        '''
        分别对 pt 格式和 onnx 格式的模型进行推理
        '''
        # pt 模型推理
        if self.pt_or_onnx == 'pt':
            model = self.model

            # 设置设备
            device = self.device
            if device.type == 'cuda':
                print('正在使用 Nvidia GPU 进行推理')
            else:
                print('正在使用 CPU 进行推理')

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
            df['class_name'] = df['class'].apply(lambda x: self.class_names[int(x)])

            # 打印感兴趣的列
            print("\nPyTorch 推理结果:")
            print(df[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

            # 绘制检测框
            img_draw = self.draw_detections(img.copy(), df)

            # 保存检测结果图像 (可选)
            if save:
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_draw)
                print(f"检测结果已保存至: {save_path}")

            # 显示检测结果图像
            cv2.imshow('PyTorch Detections', img_draw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # onnx 模型推理
        elif self.pt_or_onnx == 'onnx':
            # 预处理图像
            original_img = cv2.imread(self.image_path)
            preprocessed_img, ratio, padding = self.preprocess_image(original_img)
            img = preprocessed_img

            # 进行推理
            outputs = self.session.run(self.output_names, {self.input_name: img})[0]

            # 后处理
            df = self.postprocess_image(outputs, ratio, padding)

            # 打印检测结果
            print("\nONNX 推理结果:")
            print(df[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

            # 绘制检测框
            detected_img = self.draw_detections(original_img.copy(), df)

            # 保存结果（可选）
            if save:
                cv2.imwrite(save_path, detected_img)
                print(f"检测结果已保存至: {save_path}")

            # 显示结果
            cv2.imshow('ONNX Detections', detected_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # 推理函数（基于视频输入）
    def video_inference(self, save=False, save_path='output_video.avi', fps=30):
        '''
        对视频进行推理
        参数:
            save (bool): 是否保存检测结果视频
            save_path (str): 保存视频的路径
            fps (int): 保存视频的帧率
        '''
        # 打开视频文件或摄像头
        if self.video_source is None or self.video_source == '':
            print("未指定视频路径，尝试打开摄像头...")
            cap = cv2.VideoCapture(0)  # 打开默认摄像头
        else:
            if not pathlib.Path(self.video_source).exists():
                print(f"视频文件不存在: {self.video_source}")
                sys.exit(1)
            cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            print("无法打开视频源")
            sys.exit(1)

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps > 0:
            fps = original_fps

        # 初始化视频写入器
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"检测结果视频将保存至: {save_path}")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束

            frame_count += 1
            print(f"处理第 {frame_count} 帧...", end='\r')

            # 根据模型类型进行预处理和推理
            if self.pt_or_onnx == 'pt':
                # 直接传递帧作为 numpy 数组
                results = self.model(frame)

                # 将检测结果转换为 Pandas DataFrame
                df = results.pandas().xyxy[0]

                # 添加类别名称
                df['class_name'] = df['class'].apply(lambda x: self.class_names[int(x)])

            elif self.pt_or_onnx == 'onnx':
                # 预处理
                preprocessed_img, original_img, ratio, padding = self.preprocess_image(frame)

                # 进行推理
                outputs = self.session.run(self.output_names, {self.input_name: preprocessed_img})[0]

                # 后处理
                df = self.postprocess_image(outputs, ratio, padding)

            # 绘制检测框
            detected_frame = self.draw_detections(frame.copy(), df)

            # 显示检测结果
            cv2.imshow('Detections', detected_frame)

            # 写入视频
            if save:
                out.write(detected_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        if save:
            out.release()
        cv2.destroyAllWindows()
        print("\n视频推理完成")