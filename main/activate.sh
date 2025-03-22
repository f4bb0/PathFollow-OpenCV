#!/bin/bash

# 定义虚拟环境名称
ENV_NAME="opencv_env"

# 打印字符画

echo -e "\033[36m
 _                             __  __                     
| |     __ _  ___   ___  _ __ |  \/  |  ___  _ __   _   _ 
| |    / _\` |/ __| / _ \| '__|| |\/| | / _ \| '_ \ | | | |
| |___| (_| |\__ \|  __/| |   | |  | ||  __/| | | || |_| |
|_____|\__,_||___/ \___||_|   |_|  |_| \___||_| |_| \__,_|
\033[0m"


# 激活虚拟环境
echo "Activating Conda virtual environment: $ENV_NAME"
source activate $ENV_NAME

# 打印选择提示
echo "Which script would you like to run?"
echo "1: Rectangle.py"
echo "2: Circular.py"
echo "3: xbox.py"

# 读取用户输入
read -p "Enter your choice (1 ~ 3): " choice

# 根据用户选择运行相应脚本
if [ "$choice" == "1" ]; then
    echo "Running Rectangle.py..."
    python Rectangle.py
elif [ "$choice" == "2" ]; then
    echo "Running Circular.py..."
    python Circular.py
elif [ "$choice" == "3" ]; then
    echo "Running xbox.py..."
    python xbox.py
else
    echo "Invalid choice. Please run the script again and select 1 or 2."
fi
