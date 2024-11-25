import os
import sys
import torch
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path

import pathlib

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

class YoloComparison:
    def __init__(self, yolov5_path: str, weight_path_pt: str, weight_path_onnx: str, image_path: str):
        self.yolov5_path = yolov5_path
        self.weight_path_pt = pathlib.Path(weight_path_pt)
        self.weight_path_onnx = pathlib.Path(weight_path_onnx)
        self.image_path = image_path

        # 替换 WindowsPath 为 PosixPath
        if not pathlib.WindowsPath == pathlib.PosixPath:
            pathlib.WindowsPath = pathlib.PosixPath

        # Load PyTorch model
        if not self.weight_path_pt.exists():
            print(f"PyTorch模型文件不存在: {self.weight_path_pt}")
            sys.exit(1)
        self.device_pt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备 (PyTorch): {self.device_pt}")

        # 强制重新加载模型以避免缓存问题
        self.model_pt = torch.hub.load(
            self.yolov5_path,
            'custom',
            path=str(self.weight_path_pt),
            source='local',
            force_reload=True
        )
        self.model_pt.to(self.device_pt)
        self.model_pt.eval()
        print("已加载 PyTorch 模型")

        # Load ONNX model
        if not self.weight_path_onnx.exists():
            print(f"ONNX模型文件不存在: {self.weight_path_onnx}")
            sys.exit(1)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session_onnx = ort.InferenceSession(str(self.weight_path_onnx), providers=providers)
        self.input_name_onnx = self.session_onnx.get_inputs()[0].name
        self.output_names_onnx = [output.name for output in self.session_onnx.get_outputs()]
        print("已加载 ONNX 模型")

        # Load class names
        self.class_names = self.load_class_names()
        print("\n类别映射:")
        for class_id, class_name in enumerate(self.class_names):
            print(f"Class ID: {class_id}, Class Name: {class_name}")

    def load_class_names(self):
        """
        加载类别名称列表。
        """
        # 根据您的数据集修改类别名称
        class_names = ['normal', 'down']
        return class_names

    def preprocess_image(self, input_size=640):
        """
        预处理输入图像，确保 PyTorch 和 ONNX 模型使用相同的预处理步骤。
        参数:
            input_size (int): 模型输入大小 (默认 640x640)。
        返回:
            img_expanded (numpy.ndarray): 经过预处理的图像。
            img (numpy.ndarray): 原始图像，用于可视化。
            ratio (tuple): 宽高缩放比例 (width_ratio, height_ratio)。
            padding (tuple): 图像填充的边距 (dw, dh)。
        """
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

    def run_pt_inference(self, img, ratio, padding, input_size=640):
        img_tensor = torch.from_numpy(img).to(self.device_pt)
        with torch.no_grad():
            results = self.model_pt(img_tensor)  # 原始输出为张量

        # 假设 PyTorch 模型的输出为 [batch_size, num_predictions, 6] 或类似格式
        outputs_pt = results[0].cpu().numpy()  # 提取第一个 batch 的结果

        # 提取信息 (假设输出为 [x_center, y_center, width, height, confidence, class])
        boxes_xywh = outputs_pt[:, :4]  # [x_center, y_center, width, height]
        confidences = outputs_pt[:, 4]  # 置信度
        class_ids = outputs_pt[:, 5].astype(int)  # 类别 ID

        # 转换坐标为 [x_min, y_min, x_max, y_max]
        boxes_xyxy = self.xywh_to_xyxy(boxes_xywh)

        # 应用 NMS
        conf_threshold = 0.5
        iou_threshold = 0.45
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
        df_pt = pd.DataFrame({
            'class_id': class_ids,
            'confidence': confidences,
            'xmin': boxes_xyxy[:, 0],
            'ymin': boxes_xyxy[:, 1],
            'xmax': boxes_xyxy[:, 2],
            'ymax': boxes_xyxy[:, 3]
        })
        df_pt['class_name'] = df_pt['class_id'].apply(lambda x: self.class_names[x] if x < len(self.class_names) else str(x))

        return df_pt

    def run_onnx_inference(self, img, ratio, padding, input_size=640):
        """
        使用 ONNX 模型进行推理，并将检测框坐标还原到原始图像尺寸，应用 NMS。
        """
        img = img.astype(np.float32)

        # 运行推理
        outputs = self.session_onnx.run(self.output_names_onnx, {self.input_name_onnx: img})[0]
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
        conf_threshold = 0.5
        iou_threshold = 0.45
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

    def compare_models(self):
        """
        比较 PyTorch 和 ONNX 模型的推理结果。
        """
        # 预处理图像
        img_preprocessed, original_img, ratio, padding = self.preprocess_image()

        # PyTorch 推理
        df_pt = self.run_pt_inference(img_preprocessed, ratio, padding)
        print("\nPyTorch 推理结果:")
        print(df_pt[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

        # ONNX 推理
        df_onnx = self.run_onnx_inference(img_preprocessed, ratio, padding)
        print("\nONNX 推理结果:")
        print(df_onnx[['class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

        # 绘图
        img_with_pt = self.draw_detections_pt(original_img.copy(), df_pt)
        img_with_onnx = self.draw_detections_onnx(original_img.copy(), df_onnx)

        # 显示结果
        cv2.imshow('PyTorch Detections', img_with_pt)
        cv2.imshow('ONNX Detections', img_with_onnx)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_detections_pt(self, img, df_pt):
        """
        在图像上绘制PyTorch的检测框。
        """
        for _, row in df_pt.iterrows():
            x_min = int(row['xmin'])
            y_min = int(row['ymin'])
            x_max = int(row['xmax'])
            y_max = int(row['ymax'])

            # 绘制边界框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 绘制标签
            label = f"PT: {row['class_name']} {row['confidence']:.2f}"
            cv2.putText(img, label, (x_min, y_min + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

    def draw_detections_onnx(self, img, df_onnx):
        """
        在图像上绘制ONNX的检测框。
        参数:
            img: 原始图像
            df_onnx: ONNX推理结果的DataFrame
            input_size: 模型输入尺寸 (默认640)
        """

        # 绘制ONNX检测框
        for _, row in df_onnx.iterrows():
            x_min = int(row['xmin'])
            y_min = int(row['ymin'])
            x_max = int(row['xmax'])
            y_max = int(row['ymax'])

            # 绘制边界框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # 绘制标签
            label = f"ONNX: {row['class_name']} {row['confidence']:.2f}"
            cv2.putText(img, label, (x_min, y_min + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return img

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="比较PyTorch和ONNX模型的推理结果")
    parser.add_argument('--yolov5_path', type=str, required=True, help='YOLOv5本地路径')
    parser.add_argument('--weight_path_pt', type=str, required=True, help='PyTorch模型权重路径 (.pt)')
    parser.add_argument('--weight_path_onnx', type=str, required=True, help='ONNX模型权重路径 (.onnx)')
    parser.add_argument('--image_path', type=str, required=True, help='待检测的图像路径')

    args = parser.parse_args()

    # 初始化比较器
    comparator = YoloComparison(
        yolov5_path=args.yolov5_path,
        weight_path_pt=args.weight_path_pt,
        weight_path_onnx=args.weight_path_onnx,
        image_path=args.image_path
    )

    # 进行比较
    comparator.compare_models()