# 基于 YOLOv5 的跌倒检测器

这是一个基于 **YOLOv5** 的跌倒检测器项目。该项目利用深度学习模型实现对视频或图片中的跌倒事件进行实时检测，支持 PyTorch 和 ONNX 格式的模型加载

## 目录

- [基于 YOLOv5 的跌倒检测器](#基于-yolov5-的跌倒检测器)
  - [目录](#目录)
  - [主要组成](#主要组成)
    - [1. YOLOv5 仓库](#1-yolov5-仓库)
    - [2. 本地推理（inferlocal）](#2-本地推理inferlocal)
    - [3. 远程推理（inferemote）](#3-远程推理inferemote)
    - [4. 模型参数 (modelweight)](#4-模型参数-modelweight)
    - [5. 远程推理依赖（requirements）](#5-远程推理依赖requirements)
  - [安装指南](#安装指南)
  - [使用指南](#使用指南)
    - [命令行用法](#命令行用法)
      - [命令行参数说明](#命令行参数说明)
      - [示例用法](#示例用法)
    - [API 使用](#api-使用)
      - [示例代码](#示例代码)
    - [远程推理](#远程推理)
      - [示例用法](#示例用法-1)
  
## 主要组成
### 1. YOLOv5 仓库

本项目依赖于 [YOLOv5](https://github.com/ultralytics/yolov5) 仓库，用于模型的训练和推理。

### 2. 本地推理（inferlocal）

本地推理包含了 `FallDownDetectYolo` 模块，支持命令行运行的`run.py` 脚本以及 `run_api.py` 的api调用示例。

### 3. 远程推理（inferemote）

远程推理包含了 `fall.py` 模块和 `test.py` 脚本，可以在本地进行端口转发，并在Atlas200DK开发板上挂载om模型后实行远程推理。

### 4. 模型参数 (modelweight)

包含了训练好的`pt`、`onnx`以及`om`文件。

### 5. 远程推理依赖（requirements）

包含远程推理需要的 `inferemote` 软件包。

## 安装指南

1. **克隆项目仓库**

    ```bash
    git clone https://github.com/SowingG2333/fall-detection.git
    ```

2. **创建虚拟环境（可选）**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **安装YOLOv5依赖**
   
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt
    ```

4. **安装本地推理依赖**

    ```bash
    pip install torch torchvision onnxruntime opencv-python numpy pandas argparse
    ```

5. **安装远程推理依赖（基于Atlas200DK开发板，可选）**

    ```bash
    # windows安装
    pip install requirements/inferemote-2.0.2-py39-none-win_amd64
    # macos安装
    pip install requirements/inferemote-2.0.2-py39-none-macosx_10_15_universal2
    ```

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

### 远程推理
#### 示例用法
1. 利用ssh工具与远程开发板进行端口转发（端口可更改）
```bash
ssh -L 9023:localhost:9666 username@ip -p 9023
```
2. 利用airloader工具进行om模型挂载
```bash
airloader -m /path/to/model.om -p 9666
```
3. 本地运行 `test.py` 脚本
```bash
python test.py -r localhost -p 9023 -w 5    
```