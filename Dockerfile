# Bittle四足歩行ロボット深層強化学習プロジェクト用Dockerfile
# NVIDIA CUDA 12.8.1対応のベースイメージを使用
FROM ubuntu:24.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/bin:/usr/local/bin:${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64
ENV CUDA_VISIBLE_DEVICES=0

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libosmesa6-dev \
    patchelf \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CUDA Toolkit 12.1のインストール
RUN wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run && \
    sh cuda_12.1.1_530.30.02_linux.run --silent --toolkit && \
    rm cuda_12.1.1_530.30.02_linux.run

# Pythonエイリアスの設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# シンボリックリンクの作成（確実性を高める）
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch（CUDA対応）のインストール
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# プロジェクトの作業ディレクトリを設定
WORKDIR /app

# 主要な依存関係を個別にインストール（互換性確保）
RUN pip install gymnasium pybullet stable-baselines3 sb3-contrib \
    scipy matplotlib seaborn pandas pyyaml tqdm rich tensorboard \
    hydra-core jupyter jupyterlab ipywidgets \
    opencv-python plotly dash imageio imageio-ffmpeg \
    urdfpy trimesh numba psutil \
    --break-system-packages

# requirements.txtをコピー（参考用）
COPY requirements.txt /app/

# プロジェクトのソースコードをコピー
COPY . /app/

# Jupyter Notebookの設定
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# PyBulletのGUI表示用設定（Xvfb使用）
ENV DISPLAY=:99

# ポートの公開
EXPOSE 8888 6006

# 起動スクリプト
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# デフォルトコマンド
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
