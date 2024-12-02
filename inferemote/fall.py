import numpy as np
import cv2 as cv
import platform
import pathlib
from inferemote.atlas_remote import AtlasRemote

class FallDownDetectYolo(AtlasRemote):
    MODEL_WIDTH = 640
    MODEL_HEIGHT = 640

    def __init__(self, **kwargs):
        super().__init__(port=5562, **kwargs)
        # 检测操作系统类型保证模型路径的兼容性
        if platform.system() != "Windows":
            pathlib.WindowsPath = pathlib.PosixPath

        self.ratio = None
        self.padding = None

        # 加载类别名称
        self.class_names = self.load_class_names()

    def load_class_names(self):
        """
        加载类别名称，0对应 normal ，1对应 down
        """
        class_names = ["normal", "down"]  # 根据您的实际类别修改
        return class_names

    def pre_process(self, frame):
        """
        图像预处理，包括调整大小、填充、颜色空间转换、归一化
        """
        # 将BGR转换为RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h = image.shape[0]
        img_w = image.shape[1]

        # 目标网络输入尺寸
        net_h = self.MODEL_HEIGHT
        net_w = self.MODEL_WIDTH

        # 计算缩放比例
        scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # 计算填充位置
        shift_x = (net_w - new_w) // 2
        shift_y = (net_h - new_h) // 2

        # 调整图像大小
        tmp_image = cv.resize(image, (new_w, new_h))

        # 创建新的图像并进行填充
        new_image = np.zeros((net_h, net_w, 3), np.uint8)
        new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(tmp_image)

        # 归一化（将像素值缩放至 [0, 1] 区间）
        new_image = new_image.astype(np.float32)
        new_image = new_image / 255.0

        # 转换维度顺序
        new_image = np.transpose(new_image, (2, 0, 1))

        # 计算 ratio 和 padding
        self.ratio = (scale, scale)  # scale 在 x 和 y 方向上相同
        self.padding = (shift_x, shift_y)

        # 返回处理后的图像
        return new_image.tobytes()

    def post_process(self, result):
        """
        后处理结果，包括解码边界框和应用NMS
        """
        # 解码：从 result[0] 中获取输出数据
        blob = np.frombuffer(result[0], np.float32)  # 读取并解码为 float32 数组
        blob = blob.reshape(-1, 7)  # 重塑为 [N, 7] 形状

        # 提取坐标 (x_center, y_center, width, height)、置信度和类别概率
        boxes_xywh = blob[:, :4]  # [x_center, y_center, width, height]
        confidences = blob[:, 4]  # 置信度
        class_scores = blob[:, 5:]  # 每个类别的概率

        # 获取最高类别和置信度
        class_ids = np.argmax(class_scores, axis=1)  # 获取每个框的最高类别
        confidences = confidences * class_scores[np.arange(len(class_scores)), class_ids]  # 计算最终置信度

        # 转换为 [x_min, y_min, x_max, y_max] 格式
        boxes_xyxy = self.xywh_to_xyxy(boxes_xywh)

        # 应用 NMS（非极大值抑制），选出置信度最高的框并进行去重
        indices = cv.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confidences.tolist(),
            score_threshold=0.55,  # 置信度阈值
            nms_threshold=0.45  # NMS 阈值
        )

        if len(indices) > 0:
            indices = indices.flatten()  # 展平
            boxes_xyxy = boxes_xyxy[indices]
            confidences = confidences[indices]
            class_ids = class_ids[indices]
        else:
            print("未检测到目标")
            return []  # 如果没有检测到目标，返回空列表

        # 坐标还原
        # 假设 padding 和 ratio 由外部传入，用来还原到原始图像尺寸
        boxes_xyxy[:, [0, 2]] -= self.padding[0]  # 去掉横向填充
        boxes_xyxy[:, [1, 3]] -= self.padding[1]  # 去掉纵向填充
        boxes_xyxy[:, [0, 2]] /= self.ratio[0]  # 恢复宽度比例
        boxes_xyxy[:, [1, 3]] /= self.ratio[1]  # 恢复高度比例

        # 返回处理后的结果，返回一个包含检测信息的列表
        result_list = []
        for i in range(len(boxes_xyxy)):
            result_list.append({
                'class_id': class_ids[i],
                'confidence': confidences[i],
                'xmin': boxes_xyxy[i, 0],
                'ymin': boxes_xyxy[i, 1],
                'xmax': boxes_xyxy[i, 2],
                'ymax': boxes_xyxy[i, 3],
                'class_name': self.class_names[class_ids[i]]
            })

        print("检测结果:", result_list)

        return result_list

    def xywh_to_xyxy(self, boxes):
        """
        将 [x_center, y_center, width, height] 格式转换为 [x_min, y_min, x_max, y_max] 格式
        """
        x_center, y_center, width, height = (
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2],
            boxes[:, 3],
        )
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        return np.stack([x_min, y_min, x_max, y_max], axis=1)