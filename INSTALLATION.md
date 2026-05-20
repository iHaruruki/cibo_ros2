# インストール方法 (Installation Guide)

[日本語](#日本語-インストール方法) | [English](#english-installation-guide)

---

## 日本語 インストール方法

### 前提条件

- **OS**: Ubuntu 20.04 LTS 以上 (Ubuntu 22.04 推奨)
- **ROS2**: Humble 以上がインストール済み
- **Python**: 3.8 以上
- インターネット接続

### ステップ 1: ROS2 の基本インストール

まず、ROS2 がインストールされていない場合はインストールします。

```bash
# ROS2 Humble の場合
sudo apt update
sudo apt install -y curl gnupg2 lsb-release ubuntu-keyring
curl -sSL https://repo.ros2.org/ros.key | sudo apt-key add -
sudo add-apt-repository "deb [arch=$(dpkg --print-architecture)] http://repo.ros2.org/ubuntu $(lsb_release -cs) main"
sudo apt update
sudo apt install -y ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### ステップ 2: ビルドツールのインストール

```bash
# 基本的な開発ツール
sudo apt install -y build-essential cmake git wget curl

# colcon ビルドツール
sudo apt install -y python3-colcon-common-extensions
```

### ステップ 3: ROS2 パッケージの依存関係をインストール

```bash
# 以下のコマンドで必要な ROS2 パッケージをインストール
export ROS_DISTRO=humble
sudo apt install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-image-transport \
  ros-${ROS_DISTRO}-image-transport-plugins \
  ros-${ROS_DISTRO}-message-filters \
  ros-${ROS_DISTRO}-tf2-ros \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-${ROS_DISTRO}-std-msgs \
  ros-${ROS_DISTRO}-geometry-msgs \
  ros-${ROS_DISTRO}-xacro \
  ros-${ROS_DISTRO}-urdf-tutorial \
  ros-${ROS_DISTRO}-robot-state-publisher \
  ros-${ROS_DISTRO}-rviz2

# オプション: 画像圧縮トランスポート
sudo apt install -y \
  ros-${ROS_DISTRO}-theora-image-transport \
  ros-${ROS_DISTRO}-ffmpeg-image-transport \
  ros-${ROS_DISTRO}-ffmpeg-image-transport-tools
```

### ステップ 4: OpenCV のインストール

```bash
# システムパッケージ経由
sudo apt install -y libopencv-dev python3-opencv

# または pip 経由（最新バージョン）
pip3 install opencv-python opencv-contrib-python
```

### ステップ 5: cibo リポジトリのクローンとビルド

```bash
# ワークスペースの作成
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# cibo リポジトリのクローン
git clone https://github.com/iHaruruki/cibo.git

# ワークスペースをビルド
cd ~/ros2_ws
colcon build --symlink-install --packages-select cibo

# セットアップを読み込む
source install/setup.bash
```

---

## MediaPipe のインストール

現在の C++ 版 cibo では、MediaPipe の pose/hand/face detection 機能は含まれていません。以下の方法でこの機能を追加できます。

### オプション 1: Python ノードを使用（推奨）

Python 版の MediaPipe ノードと C++ 版の処理ノードを並行して使用します。

#### 1.1 MediaPipe ライブラリのインストール

```bash
# MediaPipe のインストール
pip3 install -U "numpy==1.26.4" "opencv-python==4.10.0.84"
pip3 install mediapipe

# 確認
python3 -c "import mediapipe; print(mediapipe.__version__)"
```

#### 1.2 Python ノードの実装

元の Python ノード（top_camera.py など）をカスタマイズして使用します。

```bash
# Python パッケージのセットアップ（別プロジェクトの場合）
cd ~/ros2_ws/src
# Python ベースの cibo パッケージを作成
colcon build --packages-select <python-package-name>
```

### オプション 2: TensorFlow Lite を使用

MediaPipe の軽量な代替として TensorFlow Lite を使用します。

```bash
# TensorFlow Lite のインストール
pip3 install tensorflow-lite

# OpenCV と組み合わせて使用
pip3 install mediapipe-lite  # 軽量版（利用可能な場合）
```

### オプション 3: OpenVINO を使用

Intel の OpenVINO を使用した推論（高速）

```bash
# OpenVINO のインストール
pip3 install openvino openvino-dev

# OpenCV との連携
sudo apt install -y libopenvino-dev libopenvino-c-dev
```

### オプション 4: ONNX Runtime を使用

ONNX 形式のモデルを実行

```bash
# ONNX Runtime のインストール
pip3 install onnxruntime

# MediaPipe モデルを ONNX 形式に変換して使用
# https://github.com/google/mediapipe/tree/master/mediapipe/tasks/cc/vision/pose_detector
```

---

## トラブルシューティング

### 問題 1: `colcon build` が失敗する

```bash
# ビルドキャッシュをクリア
cd ~/ros2_ws
rm -rf build install log
colcon build --symlink-install --packages-select cibo --verbose
```

### 問題 2: ROS2 パッケージが見つからない

```bash
# ROS_DISTRO が正しく設定されているか確認
echo $ROS_DISTRO

# setup.bash が読み込まれているか確認
source /opt/ros/$ROS_DISTRO/setup.bash
```

### 問題 3: OpenCV が見つからない

```bash
# pkg-config を使用して確認
pkg-config --cflags --libs opencv4

# または CMake で手動指定
colcon build --symlink-install --packages-select cibo \
  -DCMAKE_PREFIX_PATH=/usr/lib/cmake/opencv4
```

### 問題 4: MediaPipe の import エラー

```bash
# Python パス確認
python3 -c "import sys; print(sys.path)"

# MediaPipe の確認
python3 -c "import mediapipe as mp; print(mp.__version__)"

# 必要に応じて再インストール
pip3 install --upgrade --force-reinstall mediapipe
```

---

## 環境構築の確認

すべてがインストールされたか確認するスクリプト:

```bash
#!/bin/bash

echo "=== ROS2 環境確認 ==="
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"

echo -e "\n=== ROS2 パッケージ確認 ==="
ros2 pkg list | grep -E "(cv_bridge|image_transport|message_filters|tf2|sensor_msgs)" || echo "Warning: Some packages not found"

echo -e "\n=== Python パッケージ確認 ==="
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')" 2>/dev/null || echo "MediaPipe: not installed"

echo -e "\n=== ビルドツール確認 ==="
cmake --version | head -1
colcon --version || echo "colcon: not found"

echo -e "\n=== cibo パッケージ確認 ==="
cd ~/ros2_ws
if [ -f "install/cibo/share/cibo/CMakeLists.txt" ]; then
    echo "cibo: ビルド済み"
else
    echo "cibo: 未ビルド"
fi
```

---

## 推奨される環境セットアップ

### ~/.bashrc に追加

```bash
# ROS2 環境
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 便利な alias
alias ros2_build='cd ~/ros2_ws && colcon build --symlink-install'
alias ros2_source='source ~/ros2_ws/install/setup.bash'
alias ros2_launch_cibo='ros2 launch cibo cibo.launch.py'
```

---

## 追加リソース

- **ROS2 公式ドキュメント**: https://docs.ros.org/en/humble/
- **MediaPipe 公式**: https://developers.google.com/mediapipe
- **OpenCV ドキュメント**: https://docs.opencv.org/
- **colcon ビルドツール**: https://colcon.readthedocs.io/

---

---

## English Installation Guide

### Prerequisites

- **OS**: Ubuntu 20.04 LTS or later (Ubuntu 22.04 recommended)
- **ROS2**: Humble or later already installed
- **Python**: 3.8 or later
- Internet connection

### Step 1: ROS2 Basic Installation

First, install ROS2 if not already installed.

```bash
# For ROS2 Humble
sudo apt update
sudo apt install -y curl gnupg2 lsb-release ubuntu-keyring
curl -sSL https://repo.ros2.org/ros.key | sudo apt-key add -
sudo add-apt-repository "deb [arch=$(dpkg --print-architecture)] http://repo.ros2.org/ubuntu $(lsb_release -cs) main"
sudo apt update
sudo apt install -y ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### Step 2: Install Build Tools

```bash
# Basic development tools
sudo apt install -y build-essential cmake git wget curl

# colcon build tool
sudo apt install -y python3-colcon-common-extensions
```

### Step 3: Install ROS2 Package Dependencies

```bash
# Install required ROS2 packages
export ROS_DISTRO=humble
sudo apt install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-image-transport \
  ros-${ROS_DISTRO}-image-transport-plugins \
  ros-${ROS_DISTRO}-message-filters \
  ros-${ROS_DISTRO}-tf2-ros \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-${ROS_DISTRO}-std-msgs \
  ros-${ROS_DISTRO}-geometry-msgs \
  ros-${ROS_DISTRO}-xacro \
  ros-${ROS_DISTRO}-urdf-tutorial \
  ros-${ROS_DISTRO}-robot-state-publisher \
  ros-${ROS_DISTRO}-rviz2

# Optional: Image transport plugins
sudo apt install -y \
  ros-${ROS_DISTRO}-theora-image-transport \
  ros-${ROS_DISTRO}-ffmpeg-image-transport \
  ros-${ROS_DISTRO}-ffmpeg-image-transport-tools
```

### Step 4: Install OpenCV

```bash
# Via system packages
sudo apt install -y libopencv-dev python3-opencv

# Or via pip (latest version)
pip3 install opencv-python opencv-contrib-python
```

### Step 5: Clone and Build cibo Repository

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone cibo repository
git clone https://github.com/iHaruruki/cibo.git

# Build workspace
cd ~/ros2_ws
colcon build --symlink-install --packages-select cibo

# Source setup
source install/setup.bash
```

---

## MediaPipe Installation

The current C++ version of cibo does not include MediaPipe pose/hand/face detection functionality. You can add this functionality using the following methods:

### Option 1: Use Python Nodes (Recommended)

Run Python-based MediaPipe nodes alongside C++ processing nodes.

#### 1.1 Install MediaPipe Library

```bash
# Install MediaPipe
pip3 install -U "numpy==1.26.4" "opencv-python==4.10.0.84"
pip3 install mediapipe

# Verify installation
python3 -c "import mediapipe; print(mediapipe.__version__)"
```

#### 1.2 Implement Python Nodes

Use or customize original Python nodes (top_camera.py, etc.).

```bash
# Setup Python packages (if separate project)
cd ~/ros2_ws/src
# Create or clone Python-based cibo package
colcon build --packages-select <python-package-name>
```

### Option 2: Use TensorFlow Lite

Use TensorFlow Lite as a lightweight alternative to MediaPipe.

```bash
# Install TensorFlow Lite
pip3 install tensorflow-lite

# Use with OpenCV
pip3 install mediapipe-lite  # Lightweight version (if available)
```

### Option 3: Use OpenVINO

Use Intel's OpenVINO for fast inference.

```bash
# Install OpenVINO
pip3 install openvino openvino-dev

# Link with OpenCV
sudo apt install -y libopenvino-dev libopenvino-c-dev
```

### Option 4: Use ONNX Runtime

Execute ONNX format models.

```bash
# Install ONNX Runtime
pip3 install onnxruntime

# Convert MediaPipe models to ONNX format
# https://github.com/google/mediapipe/tree/master/mediapipe/tasks/cc/vision/pose_detector
```

---

## Troubleshooting

### Issue 1: `colcon build` Fails

```bash
# Clear build cache
cd ~/ros2_ws
rm -rf build install log
colcon build --symlink-install --packages-select cibo --verbose
```

### Issue 2: ROS2 Package Not Found

```bash
# Check ROS_DISTRO is set correctly
echo $ROS_DISTRO

# Verify setup.bash is sourced
source /opt/ros/$ROS_DISTRO/setup.bash
```

### Issue 3: OpenCV Not Found

```bash
# Check with pkg-config
pkg-config --cflags --libs opencv4

# Or manually specify in CMake
colcon build --symlink-install --packages-select cibo \
  -DCMAKE_PREFIX_PATH=/usr/lib/cmake/opencv4
```

### Issue 4: MediaPipe Import Error

```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Verify MediaPipe
python3 -c "import mediapipe as mp; print(mp.__version__)"

# Reinstall if necessary
pip3 install --upgrade --force-reinstall mediapipe
```

---

## Verify Installation

Script to verify all installations:

```bash
#!/bin/bash

echo "=== ROS2 Environment Check ==="
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"

echo -e "\n=== ROS2 Packages Check ==="
ros2 pkg list | grep -E "(cv_bridge|image_transport|message_filters|tf2|sensor_msgs)" || echo "Warning: Some packages not found"

echo -e "\n=== Python Packages Check ==="
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')" 2>/dev/null || echo "MediaPipe: not installed"

echo -e "\n=== Build Tools Check ==="
cmake --version | head -1
colcon --version || echo "colcon: not found"

echo -e "\n=== cibo Package Check ==="
cd ~/ros2_ws
if [ -f "install/cibo/share/cibo/CMakeLists.txt" ]; then
    echo "cibo: Built"
else
    echo "cibo: Not built"
fi
```

---

## Recommended Environment Setup

### Add to ~/.bashrc

```bash
# ROS2 environment
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Useful aliases
alias ros2_build='cd ~/ros2_ws && colcon build --symlink-install'
alias ros2_source='source ~/ros2_ws/install/setup.bash'
alias ros2_launch_cibo='ros2 launch cibo cibo.launch.py'
```

---

## Additional Resources

- **ROS2 Official Docs**: https://docs.ros.org/en/humble/
- **MediaPipe Official**: https://developers.google.com/mediapipe
- **OpenCV Documentation**: https://docs.opencv.org/
- **colcon Build Tool**: https://colcon.readthedocs.io/
