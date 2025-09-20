# Bittle四足歩行ロボット深層強化学習プロジェクト仕様書

## プロジェクト概要

### 目標
Bittle四足歩行ロボットをPyBulletシミュレーション環境で深層強化学習により**基本的な前進歩行**を学習させる。

### プロジェクトの特徴
- **シンプル**: 複雑な地形適応や障害物回避は含まない
- **達成可能**: 基本的な前進歩行のみに焦点
- **段階的**: 環境→学習→評価の順で実装

## プロジェクト構造

```
walking_DRL/
├── src/
│   ├── __init__.py
│   ├── environment.py          # PyBullet環境クラス
│   ├── training.py             # 学習スクリプト
│   └── evaluation.py           # 評価スクリプト
├── configs/
│   └── config.yaml             # 設定ファイル
├── models/                     # 保存されたモデル
├── logs/                       # 学習ログ
├── notebooks/
│   ├── 01_test_environment.ipynb
│   └── 02_training_results.ipynb
├── requirements.txt
├── dockerfile
├── docker-entrypoint.sh
└── README.md
```

## 1. 環境仕様 (environment.py)

### 1.1 BittleEnvironment クラス

```python
class BittleEnvironment(gym.Env):
    def __init__(self, config):
        # PyBulletクライアント初期化
        # URDFファイル読み込み
        # 状態・行動空間定義
        # 報酬関数設定
```

### 1.2 状態空間 (Observation Space)
- **関節角度**: 8次元 (4脚 × 2関節)
- **ロボット姿勢**: 3次元 (roll, pitch, yaw)
- **ロボット速度**: 3次元 (vx, vy, vz)
- **前回アクション**: 8次元

**合計**: 22次元の連続値

### 1.3 行動空間 (Action Space)
- **関節目標角度**: 8次元 (4脚 × 2関節)
- **範囲**: [-1.57, 1.57] ラジアン (約±90度)

### 1.4 環境パラメータ
- **制御周波数**: 50Hz
- **物理シミュレーション**: 240Hz
- **エピソード長**: 500ステップ (10秒)
- **重力**: -9.81 m/s²

## 2. 報酬関数仕様

### 2.1 基本報酬関数

```python
def calculate_reward(self, state, action, next_state):
    reward = 0.0
    
    # 1. 前進速度報酬 (最重要)
    forward_velocity = next_state['velocity'][0]  # x方向速度
    reward += forward_velocity * 10.0
    
    # 2. 生存報酬
    reward += 1.0
    
    # 3. 転倒ペナルティ
    if self.is_fallen(next_state):
        reward -= 100.0
    
    # 4. エネルギー効率 (軽微)
    energy_penalty = np.sum(np.abs(action)) * 0.01
    reward -= energy_penalty
    
    return reward
```

### 2.2 転倒判定
- ロボットの高さが0.05m以下
- ロボットの傾きが45度以上

## 3. 学習仕様 (training.py)

### 3.1 使用アルゴリズム
- **PPO (Proximal Policy Optimization)**
- **理由**: 安定した学習、実装が簡単、連続制御に適している

### 3.2 ハイパーパラメータ
```yaml
algorithm: "PPO"
total_timesteps: 1000000
learning_rate: 0.0003
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
```

### 3.3 学習プロセス
1. 環境初期化
2. PPOエージェント初期化
3. 学習ループ実行
4. 定期的なモデル保存
5. TensorBoardログ出力

## 4. 設定ファイル (configs/config.yaml)

```yaml
# 環境設定
environment:
  name: "BittleWalking-v0"
  max_episode_steps: 500
  control_frequency: 50
  physics_frequency: 240
  
# 学習設定
training:
  algorithm: "PPO"
  total_timesteps: 1000000
  save_frequency: 100000
  eval_frequency: 50000
  
# ログ設定
logging:
  log_dir: "./logs"
  tensorboard: true
  verbose: 1
```

## 5. 実装ファイル仕様

### 5.1 src/environment.py
```python
import pybullet as p
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BittleEnvironment(gym.Env):
    def __init__(self, config):
        # 実装詳細...
        # 状態空間: 22次元 (位置情報は除外)
    
    def reset(self, seed=None, options=None):
        # 環境リセット
        # 初期状態返却 (位置情報なし)
    
    def step(self, action):
        # アクション実行
        # 状態更新 (位置情報は取得しない)
        # 報酬計算
        # 終了判定
    
    def render(self):
        # 可視化
    
    def close(self):
        # リソース解放
```

### 5.2 src/training.py
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import yaml

def train_agent():
    # 設定読み込み
    # 環境作成
    # エージェント初期化
    # 学習実行
    # モデル保存
```

### 5.3 src/evaluation.py
```python
def evaluate_model(model_path, num_episodes=10):
    # モデル読み込み
    # 環境作成
    # 評価実行
    # 結果出力
```

## 6. 実装順序

### Phase 1: 環境実装
1. `src/environment.py` の実装
2. 基本的なPyBullet環境の構築
3. 状態・行動空間の定義
4. 基本報酬関数の実装

### Phase 2: 学習実装
1. `src/training.py` の実装
2. PPOエージェントの統合
3. 学習ループの構築
4. ログ・可視化の追加

### Phase 3: 評価・可視化
1. `src/evaluation.py` の実装
2. Jupyter Notebookの作成
3. 結果可視化の実装

## 7. 成功指標

### 7.1 学習目標
- **平均報酬**: 100以上
- **前進速度**: 0.1 m/s以上
- **生存率**: 80%以上（10秒間転倒しない）

### 7.2 評価方法
- 10エピソードの平均性能
- 学習曲線の可視化
- 歩行アニメーションの確認

## 8. 技術的制約

### 8.1 計算リソース
- **GPU**: CUDA対応（推奨）
- **メモリ**: 8GB以上
- **学習時間**: 2-4時間（GPU使用時）

### 8.2 依存関係
- PyBullet
- Stable-Baselines3
- Gymnasium
- PyTorch
- NumPy
- Matplotlib

## 9. トラブルシューティング

### 9.1 よくある問題
- URDFファイルのパス問題
- PyBulletの初期化エラー
- メモリ不足
- 学習が収束しない

### 9.2 解決方法
- 相対パスの確認
- Docker環境の使用
- バッチサイズの調整
- 学習率の調整

## 10. 拡張可能性

### 10.1 将来の拡張
- 複数アルゴリズムの比較
- より複雑な地形での学習
- 転移学習の実装
- リアルロボットへの転用

### 10.2 コードの保守性
- モジュール化された設計
- 設定ファイルによる柔軟性
- 詳細なコメントとドキュメント
- テストコードの追加

---

この仕様書に基づいて、段階的に実装を進めることで、確実に動作する四足歩行ロボットの深層強化学習システムを構築できます。
