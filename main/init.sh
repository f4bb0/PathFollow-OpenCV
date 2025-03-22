#!/bin/bash

# 定义虚拟环境名称
ENV_NAME="opencv_env"

# 创建 Conda 虚拟环境并指定 Python 版本
echo "Creating Conda virtual environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.8 -y

# 激活虚拟环境
echo "Activating virtual environment: $ENV_NAME"
source activate $ENV_NAME

# 使用 pip 安装所需的库
echo "Installing packages: opencv-python, pyserial, pygame"
pip install opencv-python pyserial pygame

# 打印安装完成信息
echo "Setup completed. Virtual environment '$ENV_NAME' is ready with Python 3.8 and required packages."
