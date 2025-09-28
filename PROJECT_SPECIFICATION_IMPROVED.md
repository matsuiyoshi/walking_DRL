# Bittle四足歩行ロボット深層強化学習プロジェクト仕様書（改善版）

## プロジェクト概要

### 目標
Bittle四足歩行ロボットをPyBulletシミュレーション環境で深層強化学習により**基本的な前進歩行**を学習させる。

### プロジェクトの特徴
- **シンプル**: 複雑な地形適応や障害物回避は含まない
- **達成可能**: 基本的な前進歩行のみに焦点
- **段階的**: 環境→学習→評価の順で実装
- **厳密**: 実装可能な詳細仕様

## プロジェクト構造

```
walking_DRL/
├── src/
│   ├── __init__.py
│   ├── environment.py          # PyBullet環境クラス
│   ├── training.py             # 学習スクリプト
│   ├── evaluation.py          # 評価スクリプト
│   └── utils/
│       ├── __init__.py
│       ├── config_validator.py # 設定検証
│       ├── logger.py          # ログ管理
│       └── exceptions.py      # カスタム例外
├── configs/
│   ├── default.yaml           # デフォルト設定
│   ├── training.yaml          # 学習設定
│   └── evaluation.yaml        # 評価設定
├── tests/
│   ├── __init__.py
│   ├── test_environment.py   # 環境テスト
│   ├── test_training.py       # 学習テスト
│   └── test_utils.py          # ユーティリティテスト
├── models/                    # 保存されたモデル
├── logs/                      # 学習ログ
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
import pybullet as p
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import os

class BittleEnvironment(gym.Env):
    """
    Bittle四足歩行ロボットのPyBullet環境
    
    Args:
        config (Dict): 環境設定辞書
        render_mode (Optional[str]): レンダリングモード
    """
    
    def __init__(self, config: Dict, render_mode: Optional[str] = None):
        super().__init__()
        
        # 設定の検証
        self._validate_config(config)
        
        # 環境パラメータ
        self.config = config
        self.render_mode = render_mode
        
        # PyBulletクライアント初期化
        self._initialize_physics()
        
        # ロボットモデル読み込み
        self._load_robot()
        
        # 状態・行動空間定義
        self._define_spaces()
        
        # 報酬関数設定
        self._setup_reward_function()
        
        # 内部状態
        self.episode_steps = 0
        self.last_action = np.zeros(8)
        
    def _validate_config(self, config: Dict) -> None:
        """設定の妥当性を検証"""
        required_keys = ['max_episode_steps', 'control_frequency', 'physics_frequency']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required config key '{key}' not found")
    
    def _initialize_physics(self) -> None:
        """物理エンジンの初期化"""
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # 物理パラメータ設定
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.config['physics_frequency'])
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'bittle-urdf'))
    
    def _load_robot(self) -> None:
        """ロボットモデルの読み込み"""
        urdf_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'bittle-urdf', 'bittle.urdf')
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        self.robot_id = p.loadURDF(urdf_path)
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # 関節情報の取得
        self.joint_indices = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # 回転関節のみ
                self.joint_indices.append(i)
    
    def _define_spaces(self) -> None:
        """状態・行動空間の定義"""
        # 状態空間: 22次元
        # 関節角度(8) + 姿勢(3) + 速度(3) + 前回アクション(8)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi] * 8 + [-np.pi] * 3 + [-10.0] * 3 + [-np.pi] * 8),
            high=np.array([np.pi] * 8 + [np.pi] * 3 + [10.0] * 3 + [np.pi] * 8),
            dtype=np.float32
        )
        
        # 行動空間: 8次元 (関節目標角度)
        self.action_space = spaces.Box(
            low=np.array([-1.57] * 8),  # -90度
            high=np.array([1.57] * 8),  # +90度
            dtype=np.float32
        )
    
    def _setup_reward_function(self) -> None:
        """報酬関数の設定"""
        self.reward_weights = {
            'forward_velocity': 10.0,
            'survival': 1.0,
            'fall_penalty': -100.0,
            'energy_efficiency': -0.01
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """環境のリセット"""
        super().reset(seed=seed)
        
        # 物理環境のリセット
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # ロボットの再読み込み
        self._load_robot()
        
        # 初期状態の設定
        self._set_initial_state()
        
        # 内部状態のリセット
        self.episode_steps = 0
        self.last_action = np.zeros(8)
        
        # 初期観測の取得
        observation = self._get_observation()
        info = {'episode_steps': 0}
        
        return observation, info
    
    def _set_initial_state(self) -> None:
        """初期状態の設定"""
        # 初期位置・姿勢
        initial_position = [0.0, 0.0, 0.1]  # 10cm浮上
        initial_orientation = [0.0, 0.0, 0.0]  # 水平
        
        p.resetBasePositionAndOrientation(
            self.robot_id, 
            initial_position, 
            p.getQuaternionFromEuler(initial_orientation)
        )
        
        # 初期関節角度の設定
        initial_joint_angles = [0.0] * 8
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, initial_joint_angles[i])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """環境のステップ実行"""
        # アクションの検証
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} not in action space")
        
        # 関節制御の実行
        self._apply_action(action)
        
        # 物理シミュレーションの実行
        for _ in range(int(self.config['physics_frequency'] / self.config['control_frequency'])):
            p.stepSimulation()
        
        # 状態の取得
        observation = self._get_observation()
        
        # 報酬の計算
        reward = self._calculate_reward(action)
        
        # 終了条件の判定
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # 情報の更新
        info = {
            'episode_steps': self.episode_steps,
            'forward_velocity': self._get_forward_velocity(),
            'robot_height': self._get_robot_height(),
            'robot_orientation': self._get_robot_orientation()
        }
        
        self.episode_steps += 1
        self.last_action = action.copy()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray) -> None:
        """アクションの適用"""
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=10.0  # 最大トルク
            )
    
    def _get_observation(self) -> np.ndarray:
        """観測の取得"""
        # 関節角度の取得
        joint_angles = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_angles.append(joint_state[0])
        
        # ロボットの位置・姿勢・速度の取得
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        
        # 姿勢をオイラー角に変換
        euler_angles = p.getEulerFromQuaternion(orientation)
        
        # 観測ベクトルの構築
        observation = np.concatenate([
            joint_angles,           # 8次元
            euler_angles,           # 3次元
            velocity,               # 3次元
            self.last_action        # 8次元
        ])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """報酬の計算"""
        reward = 0.0
        
        # 1. 前進速度報酬
        forward_velocity = self._get_forward_velocity()
        reward += forward_velocity * self.reward_weights['forward_velocity']
        
        # 2. 生存報酬
        reward += self.reward_weights['survival']
        
        # 3. 転倒ペナルティ
        if self._is_fallen():
            reward += self.reward_weights['fall_penalty']
        
        # 4. エネルギー効率ペナルティ
        energy_penalty = np.sum(np.abs(action)) * abs(self.reward_weights['energy_efficiency'])
        reward -= energy_penalty
        
        return reward
    
    def _get_forward_velocity(self) -> float:
        """前進速度の取得"""
        velocity, _ = p.getBaseVelocity(self.robot_id)
        return velocity[0]  # X方向の速度
    
    def _get_robot_height(self) -> float:
        """ロボットの高さの取得"""
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        return position[2]
    
    def _get_robot_orientation(self) -> np.ndarray:
        """ロボットの姿勢の取得"""
        _, orientation = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(p.getEulerFromQuaternion(orientation))
    
    def _is_fallen(self) -> bool:
        """転倒判定"""
        # 高さチェック
        height = self._get_robot_height()
        if height < 0.05:  # 5cm以下
            return True
        
        # 姿勢チェック
        orientation = self._get_robot_orientation()
        if abs(orientation[0]) > np.pi/4 or abs(orientation[1]) > np.pi/4:  # 45度以上
            return True
        
        return False
    
    def _is_terminated(self) -> bool:
        """終了条件の判定"""
        return self._is_fallen()
    
    def _is_truncated(self) -> bool:
        """切り捨て条件の判定"""
        return self.episode_steps >= self.config['max_episode_steps']
    
    def render(self) -> Optional[np.ndarray]:
        """環境の可視化"""
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        return None
    
    def close(self) -> None:
        """環境の終了"""
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
```

### 1.2 状態空間の詳細仕様

```python
# 状態空間の構成
OBSERVATION_SPACE_SPEC = {
    'joint_angles': {
        'dimension': 8,
        'range': [-np.pi, np.pi],
        'description': '4脚 × 2関節の角度'
    },
    'robot_orientation': {
        'dimension': 3,
        'range': [-np.pi, np.pi],
        'description': 'roll, pitch, yaw'
    },
    'robot_velocity': {
        'dimension': 3,
        'range': [-10.0, 10.0],
        'description': 'vx, vy, vz'
    },
    'last_action': {
        'dimension': 8,
        'range': [-np.pi, np.pi],
        'description': '前回のアクション'
    }
}
```

### 1.3 行動空間の詳細仕様

```python
# 行動空間の構成
ACTION_SPACE_SPEC = {
    'joint_target_angles': {
        'dimension': 8,
        'range': [-1.57, 1.57],  # ±90度
        'description': '4脚 × 2関節の目標角度'
    }
}
```

## 2. 報酬関数の詳細仕様

### 2.1 報酬関数の数学的定義

```python
def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    """
    報酬関数の計算
    
    Args:
        state: 現在の状態
        action: 実行したアクション
        next_state: 次の状態
    
    Returns:
        float: 報酬値
    """
    reward = 0.0
    
    # 1. 前進速度報酬 (最重要)
    forward_velocity = next_state[11]  # X方向速度
    reward += forward_velocity * 10.0
    
    # 2. 生存報酬
    reward += 1.0
    
    # 3. 転倒ペナルティ
    if self._is_fallen():
        reward -= 100.0
    
    # 4. エネルギー効率ペナルティ
    energy_penalty = np.sum(np.abs(action)) * 0.01
    reward -= energy_penalty
    
    return reward
```

### 2.2 報酬関数の重み付け根拠

```python
REWARD_WEIGHTS = {
    'forward_velocity': 10.0,    # 前進速度の重要度
    'survival': 1.0,             # 生存の基本報酬
    'fall_penalty': -100.0,      # 転倒ペナルティ
    'energy_efficiency': -0.01  # エネルギー効率
}
```

## 3. 学習仕様 (training.py)

### 3.1 学習スクリプトの詳細実装

```python
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
import tensorboard
from pathlib import Path

def train_agent(config_path: str = "configs/training.yaml"):
    """
    エージェントの学習実行
    
    Args:
        config_path: 設定ファイルのパス
    """
    # 設定の読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 環境の作成
    env = make_vec_env(
        lambda: BittleEnvironment(config['environment']),
        n_envs=config['training']['n_envs']
    )
    
    # 環境の正規化
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # エージェントの初期化
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        ent_coef=config['training']['ent_coef'],
        vf_coef=config['training']['vf_coef'],
        verbose=1,
        tensorboard_log=config['logging']['log_dir']
    )
    
    # コールバックの設定
    callbacks = []
    
    # 評価コールバック
    eval_callback = EvalCallback(
        env,
        best_model_save_path=config['training']['model_save_path'],
        log_path=config['logging']['log_dir'],
        eval_freq=config['training']['eval_frequency'],
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # チェックポイントコールバック
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_frequency'],
        save_path=config['training']['checkpoint_path']
    )
    callbacks.append(checkpoint_callback)
    
    # 学習の実行
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callbacks
    )
    
    # モデルの保存
    model.save(config['training']['final_model_path'])
    env.save(config['training']['vec_normalize_path'])
    
    return model
```

### 3.2 ハイパーパラメータの詳細設定

```yaml
# configs/training.yaml
training:
  algorithm: "PPO"
  total_timesteps: 1000000
  n_envs: 4
  n_steps: 2048
  learning_rate: 0.0003
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  save_frequency: 100000
  eval_frequency: 50000
  model_save_path: "./models/best_model"
  checkpoint_path: "./models/checkpoints"
  final_model_path: "./models/final_model"
  vec_normalize_path: "./models/vec_normalize.pkl"
```

## 4. 評価仕様 (evaluation.py)

### 4.1 評価スクリプトの詳細実装

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

def evaluate_model(model_path: str, vec_normalize_path: str, num_episodes: int = 10):
    """
    モデルの評価実行
    
    Args:
        model_path: モデルファイルのパス
        vec_normalize_path: 正規化ファイルのパス
        num_episodes: 評価エピソード数
    """
    # 環境の作成
    env = BittleEnvironment(config['environment'])
    
    # 正規化の読み込み
    env = VecNormalize.load(vec_normalize_path, env)
    
    # モデルの読み込み
    model = PPO.load(model_path)
    
    # 評価の実行
    episode_rewards = []
    episode_lengths = []
    forward_velocities = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        forward_velocities.append(info.get('forward_velocity', 0))
    
    # 結果の出力
    print(f"平均報酬: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均エピソード長: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"平均前進速度: {np.mean(forward_velocities):.2f} ± {np.std(forward_velocities):.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'forward_velocities': forward_velocities
    }
```

## 5. 設定ファイルの詳細仕様

### 5.1 環境設定 (configs/default.yaml)

```yaml
# 環境設定
environment:
  name: "BittleWalking-v0"
  max_episode_steps: 500
  control_frequency: 50
  physics_frequency: 240
  gravity: -9.81
  time_step: 0.004  # 1/240
  
# ロボット設定
robot:
  urdf_path: "assets/bittle-urdf/bittle.urdf"
  initial_position: [0.0, 0.0, 0.1]
  initial_orientation: [0.0, 0.0, 0.0]
  joint_limits: [-1.57, 1.57]
  max_torque: 10.0
  
# 報酬設定
rewards:
  forward_velocity_weight: 10.0
  survival_reward: 1.0
  fall_penalty: -100.0
  energy_efficiency_weight: -0.01
  
# 終了条件
termination:
  max_height: 0.05
  max_roll: 45.0  # 度
  max_pitch: 45.0  # 度
```

### 5.2 学習設定 (configs/training.yaml)

```yaml
# 学習設定
training:
  algorithm: "PPO"
  total_timesteps: 1000000
  n_envs: 4
  n_steps: 2048
  learning_rate: 0.0003
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  
# 保存設定
save:
  frequency: 100000
  model_path: "./models"
  checkpoint_path: "./models/checkpoints"
  vec_normalize_path: "./models/vec_normalize.pkl"
  
# 評価設定
evaluation:
  frequency: 50000
  n_eval_episodes: 5
  deterministic: true
```

## 6. テスト仕様

### 6.1 環境テスト (tests/test_environment.py)

```python
import unittest
import numpy as np
from src.environment import BittleEnvironment

class TestBittleEnvironment(unittest.TestCase):
    """BittleEnvironmentのテストクラス"""
    
    def setUp(self):
        """テストの初期化"""
        self.config = {
            'max_episode_steps': 100,
            'control_frequency': 50,
            'physics_frequency': 240
        }
        self.env = BittleEnvironment(self.config)
    
    def test_environment_initialization(self):
        """環境初期化のテスト"""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.episode_steps, 0)
    
    def test_reset(self):
        """リセットのテスト"""
        obs, info = self.env.reset()
        self.assertEqual(len(obs), 22)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
    
    def test_step(self):
        """ステップ実行のテスト"""
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertEqual(len(obs), 22)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_action_space(self):
        """行動空間のテスト"""
        self.assertEqual(self.env.action_space.shape, (8,))
        self.assertTrue(self.env.action_space.contains(np.zeros(8)))
        self.assertFalse(self.env.action_space.contains(np.ones(9)))
    
    def test_observation_space(self):
        """観測空間のテスト"""
        self.assertEqual(self.env.observation_space.shape, (22,))
        obs = self.env.observation_space.sample()
        self.assertTrue(self.env.observation_space.contains(obs))
    
    def tearDown(self):
        """テストの終了"""
        self.env.close()
```

### 6.2 学習テスト (tests/test_training.py)

```python
import unittest
from src.training import train_agent

class TestTraining(unittest.TestCase):
    """学習のテストクラス"""
    
    def test_training_config_validation(self):
        """学習設定の検証テスト"""
        # 設定ファイルの存在確認
        config_path = "configs/training.yaml"
        self.assertTrue(os.path.exists(config_path))
    
    def test_model_creation(self):
        """モデル作成のテスト"""
        # 簡単な学習テスト
        pass  # 実装時に追加
```

## 7. エラーハンドリング仕様

### 7.1 カスタム例外 (src/utils/exceptions.py)

```python
class EnvironmentError(Exception):
    """環境関連のエラー"""
    pass

class URDFLoadError(EnvironmentError):
    """URDF読み込みエラー"""
    pass

class PhysicsInitializationError(EnvironmentError):
    """物理エンジン初期化エラー"""
    pass

class ConfigValidationError(Exception):
    """設定検証エラー"""
    pass

class ModelLoadError(Exception):
    """モデル読み込みエラー"""
    pass
```

### 7.2 設定検証 (src/utils/config_validator.py)

```python
def validate_config(config: dict) -> bool:
    """設定の妥当性を検証"""
    required_keys = ['max_episode_steps', 'control_frequency', 'physics_frequency']
    
    for key in required_keys:
        if key not in config:
            raise ConfigValidationError(f"Required key '{key}' not found")
    
    if config['control_frequency'] > config['physics_frequency']:
        raise ConfigValidationError("Control frequency must be <= physics frequency")
    
    return True
```

## 8. ログ・モニタリング仕様

### 8.1 ログ管理 (src/utils/logger.py)

```python
import logging
import tensorboard
from pathlib import Path

def setup_logger(name: str, log_dir: str = "./logs") -> logging.Logger:
    """ロガーの設定"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # ファイルハンドラー
    log_file = Path(log_dir) / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger
```

### 8.2 メトリクス定義

```python
METRICS = {
    'episode_reward': 'エピソード報酬',
    'episode_length': 'エピソード長',
    'forward_velocity': '前進速度',
    'robot_height': 'ロボット高さ',
    'energy_consumption': 'エネルギー消費',
    'stability': '安定性指標'
}
```

## 9. 成功指標の詳細定義

### 9.1 学習目標

```python
SUCCESS_CRITERIA = {
    'average_reward': {
        'target': 100.0,
        'tolerance': 10.0,
        'description': '平均報酬'
    },
    'forward_velocity': {
        'target': 0.1,  # m/s
        'tolerance': 0.02,
        'description': '前進速度'
    },
    'survival_rate': {
        'target': 0.8,
        'tolerance': 0.1,
        'description': '生存率'
    },
    'stability': {
        'target': 0.9,
        'tolerance': 0.05,
        'description': '安定性指標'
    }
}
```

### 9.2 評価方法

```python
def evaluate_success(metrics: dict) -> bool:
    """成功判定"""
    for criterion, target in SUCCESS_CRITERIA.items():
        if criterion in metrics:
            if abs(metrics[criterion] - target['target']) > target['tolerance']:
                return False
    return True
```

## 10. 実装順序の詳細

### Phase 1: 環境実装 (1-2日)
1. `src/environment.py` の実装
2. 基本的なPyBullet環境の構築
3. 状態・行動空間の定義
4. 基本報酬関数の実装
5. 環境テストの作成

### Phase 2: 学習実装 (2-3日)
1. `src/training.py` の実装
2. PPOエージェントの統合
3. 学習ループの構築
4. ログ・可視化の追加
5. 学習テストの作成

### Phase 3: 評価・可視化 (1-2日)
1. `src/evaluation.py` の実装
2. Jupyter Notebookの作成
3. 結果可視化の実装
4. 性能評価の実行

## 11. 技術的制約の詳細

### 11.1 計算リソース

```python
RESOURCE_REQUIREMENTS = {
    'gpu': {
        'required': True,
        'memory': '8GB+',
        'cuda_version': '11.8+'
    },
    'cpu': {
        'cores': 4,
        'memory': '16GB+'
    },
    'storage': {
        'models': '2GB',
        'logs': '1GB',
        'total': '5GB+'
    }
}
```

### 11.2 実行時間

```python
EXECUTION_TIME = {
    'environment_setup': '5-10分',
    'training': '2-4時間',
    'evaluation': '10-20分',
    'total': '3-5時間'
}
```

---

この改善された仕様書により、実装時の曖昧さを排除し、確実に動作する四足歩行ロボットの深層強化学習システムを構築できます。
