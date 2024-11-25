# 基于 YOLOv5 的跌倒检测器项目

这是一个基于 YOLOv5 的跌倒检测器项目，最初计划用于食物检测。该项目利用深度学习模型实现对视频或图片中的跌倒事件进行实时检测，并支持 PyTorch 和 ONNX 格式的模型。

## 目录

- [主要组成](#主要组成)
  - [1. YOLOv5 仓库](#1-yolov5-仓库)
  - [2. FallDownDetectYolo 模块](#2-falldowndetectyolo-模块)
    - [letterbox 填充函数](#letterbox-填充函数)
    - [FallDownDetectYolo 类](#falldowndetectyolo-类)
- [安装指南](#安装指南)
- [使用指南](#使用指南)
  - [命令行用法](#命令行用法)
  - [API 使用](#api-使用)
  
## 主要组成

### 1. YOLOv5 仓库

本项目依赖于 [YOLOv5](https://github.com/ultralytics/yolov5) 仓库，用于模型的训练和推理。YOLOv5 是一个高效的目标检测算法，具有快速和准确的特点。

### 2. FallDownDetectYolo 模块

`FallDownDetectYolo` 模块是项目的核心组件，负责加载模型、处理输入、进行推理以及输出检测结果。

- **letterbox 填充函数**  
  `letterbox` 函数用于对输入图像进行预处理，保持图像的宽高比并填充到目标尺寸。这是 YOLOv5 特有的填充方式，以确保模型输入的一致性。  

- **FallDownDetectYolo 类**  
  `FallDownDetectYolo` 类封装了跌倒检测的主要功能，包括模型加载、图像/视频预处理、推理、后处理和结果绘制。主要包含以下方法：
  - `__init__`: 初始化检测器，加载模型。
  - `load_class_names`: 加载类别名称。
  - `xywh_to_xyxy`: 转换坐标格式。
  - `preprocess_image`: 图像预处理。
  - `postprocess_image`: 图像后处理。
  - `draw_detections`: 绘制检测结果。
  - `img_inference`: 对单张图片进行推理。
  - `video_inference`: 对视频进行推理。

## 安装指南

1. **克隆项目仓库**

    ```bash
    git clone https://github.com/SowingG2333/fall-down-detection-yolo.git
    ```

2. **创建虚拟环境（可选）**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **安装依赖**
   
    ```bash
    pip install torch torchvision onnxruntime opencv-python numpy pandas argparse
    ```

4. **下载 YOLOv5 模型**

    - 前往 [YOLOv5 Releases](https://github.com/ultralytics/yolov5/releases) 下载预训练模型，或使用自定义训练的模型。

## 使用指南

### 命令行用法

本项目提供了命令行接口，支持图像和视频的推理。以下是详细的命令行参数说明及使用示例。

#### 命令行参数说明

- `--yolov5_path`：YOLOv5 的本地路径。
- `--weight_path`：模型权重文件的路径（`.pt` 或 `.onnx`）。
- `--image_path`：待检测的图像路径。
- `--video_path`：待检测的视频路径。
- `--model_type`：模型类型，选择 `pt` 或 `onnx`。
- `--input_type`：输入类型，选择 `image` 或 `video`。
- `--save`：是否保存检测结果。
- `--save_path`：检测结果的保存路径（图片或视频）。
- `--fps`：保存视频的帧率（仅在视频模式下有效）。

#### 示例用法

1. **图像推理（默认不保存结果）**

    ```bash
    python run.py --yolov5_path /path/to/yolov5 \
                  --weight_path /path/to/model.pt \
                  --image_path /path/to/image.jpg \
                  --model_type pt \
                  --input_type image
    ```

2. **图像推理并保存检测结果**

    ```bash
    python run.py --yolov5_path /path/to/yolov5 \
                  --weight_path /path/to/model.pt \
                  --image_path /path/to/image.jpg \
                  --model_type pt \
                  --input_type image \
                  --save \
                  --save_path /path/to/output.jpg
    ```

3. **视频推理（默认不保存结果）**

    ```bash
    python run.py --yolov5_path /path/to/yolov5 \
                  --weight_path /path/to/model.onnx \
                  --video_path /path/to/video.mp4 \
                  --model_type onnx \
                  --input_type video
    ```

4. **视频推理并保存检测结果**

    ```bash
    python run.py --yolov5_path /path/to/yolov5 \
                  --weight_path /path/to/model.onnx \
                  --video_path /path/to/video.mp4 \
                  --model_type onnx \
                  --input_type video \
                  --save \
                  --save_path /path/to/output_video.avi \
                  --fps 30
    ```

5. **实时摄像头推理（不保存结果）**

    ```bash
    python run.py --yolov5_path /path/to/yolov5 \
                  --weight_path /path/to/model.onnx \
                  --model_type onnx \
                  --input_type video
    ```

    *如果不指定 `--video_path`，脚本将尝试打开默认摄像头进行实时推理。*

### API 使用

除了命令行接口，您还可以在 Python 脚本中直接调用 `FallDownDetectYolo` 类进行跌倒检测。以下是如何在代码中使用该模块的示例。

#### 示例代码

```python
from fallDownDetectYolo import FallDownDetectYolo

# 初始化检测器
detector = FallDownDetectYolo(
    yolov5_path='/path/to/yolov5',
    weight_path='/path/to/model.pt',
    image_path='/path/to/image.jpg',
    video_path=None,  # 如果不使用视频
    pt_or_onnx='pt'
)

# 对单张图片进行推理
detector.img_inference(save=True, save_path='/path/to/output.jpg')

# 对视频进行推理
detector.video_inference(save=True, save_path='/path/to/output_video.avi', fps=30)
```
