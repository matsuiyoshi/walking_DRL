"""
Bittle四足歩行ロボットのPyBullet環境
デバッグとモニタリングを重視した実装
"""

import pybullet as p
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional, List
import os
import time
from pathlib import Path

from .utils.exceptions import (
    EnvironmentError, URDFLoadError, PhysicsInitializationError,
    RobotStateError, ActionSpaceError
)
from .utils.logger import get_logger
from .utils.config_validator import validate_config


class BittleEnvironment(gym.Env):
    """
    Bittle四足歩行ロボットのPyBullet環境
    
    Features:
    - 詳細なログとデバッグ情報
    - エラーハンドリングと復旧機能
    - 状態監視とパフォーマンス計測
    - 段階的な初期化プロセス
    """
    
    def __init__(self, config: Dict, render_mode: Optional[str] = None):
        """
        環境の初期化
        
        Args:
            config: 環境設定辞書
            render_mode: レンダリングモード ("human", "rgb_array", None)
        """
        super().__init__()
        
        # ロガーの初期化
        self.logger = get_logger("environment", 
                                config.get('logging', {}).get('log_dir', './logs'),
                                config.get('logging', {}).get('debug_level', 'INFO'))
        
        self.logger.info("=== Bittle環境の初期化開始 ===")
        self.logger.debug("初期化パラメータ", {"config": config, "render_mode": render_mode})
        
        try:
            # 設定の検証と保存
            self._validate_and_store_config(config)
            
            # 環境変数の初期化
            self._initialize_variables(render_mode)
            
            # PyBullet物理エンジンの初期化
            self._initialize_physics()
            
            # ロボットモデルの読み込み
            self._load_robot()
            
            # 状態・行動空間の定義
            self._define_spaces()
            
            # 報酬関数の設定
            self._setup_reward_function()
            
            # デバッグ情報の初期化
            self._initialize_debug_info()
            
            self.logger.info("Bittle環境の初期化が完了しました")
            
        except Exception as e:
            self.logger.critical("環境初期化中にエラーが発生しました", exception=e)
            self._cleanup_on_error()
            raise
    
    def _validate_and_store_config(self, config: Dict):
        """設定の検証と保存"""
        self.logger.debug("設定の検証開始")
        
        # 設定検証
        validate_config(config)
        self.config = config
        
        # 主要設定の取得
        self.env_config = config['environment']
        self.robot_config = config.get('robot', {})
        self.reward_config = config.get('rewards', {})
        self.termination_config = config.get('termination', {})
        
        self.logger.debug("設定の検証完了", {
            "max_episode_steps": self.env_config['max_episode_steps'],
            "control_frequency": self.env_config['control_frequency'],
            "physics_frequency": self.env_config['physics_frequency']
        })
    
    def _initialize_variables(self, render_mode: Optional[str]):
        """環境変数の初期化"""
        self.render_mode = render_mode
        self.episode_steps = 0
        self.total_episodes = 0
        self.last_action = np.zeros(8)
        self.physics_client = None
        self.robot_id = None
        self.joint_indices = []
        
        # デバッグ用変数
        self.debug_info = {}
        self.performance_metrics = {}
        self.state_history = []
        self.action_history = []
        
        # 物理シミュレーション設定
        self.control_frequency = self.env_config['control_frequency']
        self.physics_frequency = self.env_config['physics_frequency']
        self.physics_steps_per_control = int(self.physics_frequency / self.control_frequency)
        
        self.logger.debug("環境変数の初期化完了", {
            "physics_steps_per_control": self.physics_steps_per_control,
            "render_mode": self.render_mode
        })
    
    def _initialize_physics(self):
        """物理エンジンの初期化"""
        self.logger.debug("PyBullet物理エンジンの初期化開始")
        
        try:
            # 接続モードの決定
            if self.render_mode == "human":
                self.physics_client = p.connect(p.GUI)
                self.logger.info("GUIモードで物理エンジンを初期化")
            else:
                self.physics_client = p.connect(p.DIRECT)
                self.logger.info("ダイレクトモードで物理エンジンを初期化")
            
            if self.physics_client < 0:
                raise PhysicsInitializationError(self.physics_client)
            
            # 物理パラメータの設定
            gravity = self.env_config.get('gravity', -9.81)
            time_step = 1.0 / self.physics_frequency
            
            p.setGravity(0, 0, gravity)
            p.setTimeStep(time_step)
            p.setRealTimeSimulation(0)  # ステップ実行モード
            
            self.logger.debug("物理パラメータ設定完了", {
                "gravity": gravity,
                "time_step": time_step,
                "physics_client": self.physics_client
            })
            
            # アセットパスの追加
            asset_path = Path(__file__).parent.parent / 'assets' / 'bittle-urdf'
            p.setAdditionalSearchPath(str(asset_path))
            self.logger.debug("アセットパス追加", {"asset_path": str(asset_path)})
            
        except Exception as e:
            self.logger.error("物理エンジン初期化エラー", exception=e)
            raise PhysicsInitializationError() from e
    
    def _load_robot(self):
        """ロボットモデルの読み込み"""
        self.logger.debug("ロボットモデルの読み込み開始")
        
        try:
            # URDFパスの解決
            urdf_path = self.robot_config.get('urdf_path', 'bittle.urdf')
            if not os.path.isabs(urdf_path):
                urdf_path = Path(__file__).parent.parent / 'assets' / 'bittle-urdf' / urdf_path
            
            self.logger.debug("URDF読み込み試行", {"urdf_path": str(urdf_path)})
            
            if not urdf_path.exists():
                raise URDFLoadError(str(urdf_path))
            
            # ロボットの読み込み
            initial_pos = self.robot_config.get('initial_position', [0.0, 0.0, 0.1])
            initial_orn = p.getQuaternionFromEuler(
                self.robot_config.get('initial_orientation', [0.0, 0.0, 0.0])
            )
            
            self.robot_id = p.loadURDF(
                str(urdf_path),
                basePosition=initial_pos,
                baseOrientation=initial_orn
            )
            
            if self.robot_id < 0:
                raise URDFLoadError(str(urdf_path), "Failed to load URDF")
            
            self.logger.info("ロボットモデルの読み込み成功", {
                "robot_id": self.robot_id,
                "initial_position": initial_pos
            })
            
            # 関節情報の取得
            self._analyze_robot_joints()
            
        except Exception as e:
            self.logger.error("ロボットモデル読み込みエラー", exception=e)
            raise URDFLoadError(str(urdf_path)) from e
    
    def _analyze_robot_joints(self):
        """ロボットの関節情報を解析"""
        self.logger.debug("ロボット関節情報の解析開始")
        
        try:
            self.num_joints = p.getNumJoints(self.robot_id)
            self.joint_indices = []
            self.joint_info = {}
            
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                joint_name = joint_info[1].decode('utf-8')
                joint_type = joint_info[2]
                
                self.joint_info[i] = {
                    'name': joint_name,
                    'type': joint_type,
                    'lower_limit': joint_info[8],
                    'upper_limit': joint_info[9],
                    'max_force': joint_info[10],
                    'max_velocity': joint_info[11]
                }
                
                # 回転関節のみを制御対象とする
                if joint_type == p.JOINT_REVOLUTE:
                    self.joint_indices.append(i)
            
            self.logger.info("関節情報解析完了", {
                "total_joints": self.num_joints,
                "controllable_joints": len(self.joint_indices),
                "joint_indices": self.joint_indices
            })
            
            # 関節制限の確認
            self._validate_joint_limits()
            
        except Exception as e:
            self.logger.error("関節情報解析エラー", exception=e)
            raise RobotStateError(self.robot_id) from e
    
    def _validate_joint_limits(self):
        """関節制限の検証"""
        joint_limits = self.robot_config.get('joint_limits', [-1.57, 1.22173])
        
        for joint_idx in self.joint_indices:
            joint_info = self.joint_info[joint_idx]
            lower_limit = joint_info['lower_limit']
            upper_limit = joint_info['upper_limit']
            
            # 関節制限が設定されている場合の検証
            if lower_limit < upper_limit:
                if joint_limits[0] < lower_limit or joint_limits[1] > upper_limit:
                    self.logger.warning("関節制限が物理制限を超えています", {
                        "joint_idx": joint_idx,
                        "joint_name": joint_info['name'],
                        "config_limits": joint_limits,
                        "physical_limits": [lower_limit, upper_limit]
                    })
    
    def _define_spaces(self):
        """状態・行動空間の定義"""
        self.logger.debug("状態・行動空間の定義開始")
        
        try:
            # 行動空間: 8次元 (関節目標角度)
            joint_limits = self.robot_config.get('joint_limits', [-1.22173, 1.22173])
            self.action_space = spaces.Box(
                low=np.array([joint_limits[0]] * 8),
                high=np.array([joint_limits[1]] * 8),
                dtype=np.float32
            )
            
            # 状態空間: 22次元
            # 関節角度(8) + 姿勢(3) + 速度(3) + 前回アクション(8)
            self.observation_space = spaces.Box(
                low=np.array([-np.pi] * 8 + [-np.pi] * 3 + [-10.0] * 3 + [-np.pi] * 8),
                high=np.array([np.pi] * 8 + [np.pi] * 3 + [10.0] * 3 + [np.pi] * 8),
                dtype=np.float32
            )
            
            self.logger.info("状態・行動空間の定義完了", {
                "action_space_shape": self.action_space.shape,
                "observation_space_shape": self.observation_space.shape,
                "joint_limits": joint_limits
            })
            
        except Exception as e:
            self.logger.error("状態・行動空間定義エラー", exception=e)
            raise EnvironmentError("Failed to define spaces") from e
    
    def _setup_reward_function(self):
        """報酬関数の設定"""
        self.logger.debug("報酬関数の設定開始")
        
        self.reward_weights = {
            # 基本報酬要素
            'forward_velocity': self.reward_config.get('forward_velocity_weight', 10.0),
            'survival': self.reward_config.get('survival_reward', 1.0),
            'fall_penalty': self.reward_config.get('fall_penalty', -100.0),
            'energy_efficiency': self.reward_config.get('energy_efficiency_weight', -0.01),
            
            # 追加報酬要素（default.yaml対応）
            'target_height': self.reward_config.get('target_height', 0.1),
            'height_stability_weight': self.reward_config.get('height_stability_weight', 5.0),
            'vertical_velocity_penalty': self.reward_config.get('vertical_velocity_penalty', 2.0),
            'distance_weight': self.reward_config.get('distance_weight', 1.0)
        }
        
        self.logger.info("報酬関数設定完了", {"reward_weights": self.reward_weights})
    
    def _initialize_debug_info(self):
        """デバッグ情報の初期化"""
        self.debug_info = {
            'initialization_time': time.time(),
            'step_count': 0,
            'episode_count': 0,
            'physics_steps': 0,
            'last_reward_breakdown': {},
            'performance_history': []
        }
        
        self.logger.debug("デバッグ情報初期化完了")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        環境のリセット
        
        Args:
            seed: 乱数シード
            options: リセットオプション
            
        Returns:
            Tuple[np.ndarray, Dict]: 初期観測とメタ情報
        """
        start_time = time.time()
        self.logger.info(f"=== エピソード {self.total_episodes + 1} 開始 ===")
        
        try:
            super().reset(seed=seed)
            
            # 物理シミュレーションのリセット
            self._reset_physics()
            
            # ロボットの再配置
            self._reset_robot_state()
            
            # 内部状態のリセット
            self._reset_internal_state()
            
            # 初期観測の取得
            observation = self._get_observation()
            
            # メタ情報の準備
            info = self._get_reset_info()
            
            reset_time = time.time() - start_time
            self.logger.info("環境リセット完了", {
                "episode": self.total_episodes + 1,
                "reset_time": f"{reset_time:.3f}s"
            })
            
            return observation, info
            
        except Exception as e:
            self.logger.error("環境リセット中にエラーが発生しました", exception=e)
            raise EnvironmentError("Reset failed") from e
    
    def _reset_physics(self):
        """物理シミュレーションのリセット"""
        self.logger.debug("物理シミュレーションリセット開始")
        
        # 全てのオブジェクトを削除
        p.resetSimulation()
        
        # 物理パラメータの再設定
        gravity = self.env_config.get('gravity', -9.81)
        p.setGravity(0, 0, gravity)
        p.setTimeStep(1.0 / self.physics_frequency)
        
        # 地面の追加（PyBulletの標準地面を使用）
        try:
            p.loadURDF("plane.urdf")
        except:
            # 標準のplane.urdfが見つからない場合は、シンプルな地面を作成
            self.logger.warning("plane.urdfが見つからないため、シンプルな地面を作成します")
            plane_shape = p.createCollisionShape(p.GEOM_PLANE)
            p.createMultiBody(0, plane_shape)
        
        self.logger.debug("物理シミュレーションリセット完了")
    
    def _reset_robot_state(self):
        """ロボット状態のリセット"""
        self.logger.debug("ロボット状態リセット開始")
        
        try:
            # ロボットの再読み込み
            self._load_robot()
            
            # 初期関節角度の設定
            initial_joint_angles = [0.0] * len(self.joint_indices)
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, initial_joint_angles[i])
            
            # 少し時間を進めて安定化
            for _ in range(10):
                p.stepSimulation()
            
            self.logger.debug("ロボット状態リセット完了", {
                "initial_joint_angles": initial_joint_angles
            })
            
        except Exception as e:
            self.logger.error("ロボット状態リセットエラー", exception=e)
            raise RobotStateError(self.robot_id) from e
    
    def _reset_internal_state(self):
        """内部状態のリセット"""
        self.episode_steps = 0
        self.last_action = np.zeros(8)
        self.total_episodes += 1
        
        # デバッグ情報のリセット
        self.debug_info['episode_count'] = self.total_episodes
        self.debug_info['step_count'] = 0
        self.debug_info['last_reward_breakdown'] = {}
        
        # 履歴のクリア（メモリ効率のため）
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-500:]
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]
    
    def _get_reset_info(self) -> Dict:
        """リセット時のメタ情報"""
        return {
            'episode': self.total_episodes,
            'episode_steps': 0,
            'robot_id': self.robot_id,
            'joint_count': len(self.joint_indices),
            'debug_mode': True
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        環境のステップ実行
        
        Args:
            action: 実行するアクション
            
        Returns:
            Tuple: 観測, 報酬, 終了フラグ, 切り捨てフラグ, メタ情報
        """
        step_start_time = time.time()
        
        try:
            # アクションの検証
            self._validate_action(action)
            
            # アクションの適用
            self._apply_action(action)
            
            # 物理シミュレーションの実行
            self._execute_physics_simulation()
            
            # 状態の取得
            observation = self._get_observation()
            
            # 報酬の計算
            reward, reward_breakdown = self._calculate_reward_detailed(action)
            
            # 終了条件の判定
            terminated = self._is_terminated()
            truncated = self._is_truncated()
            
            # メタ情報の更新
            info = self._get_step_info(reward_breakdown, step_start_time)
            
            # 内部状態の更新
            self._update_internal_state(action, reward, info)
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            self.logger.error("ステップ実行中にエラーが発生しました", exception=e)
            # エラー時は安全な状態を返す
            return self._get_safe_state(), -100.0, True, False, {'error': str(e)}
    
    def _validate_action(self, action: np.ndarray):
        """アクションの検証"""
        if not isinstance(action, np.ndarray):
            raise ActionSpaceError(action, {"expected_type": "np.ndarray"})
        
        if action.shape != (8,):
            raise ActionSpaceError(action, {"expected_shape": (8,), "actual_shape": action.shape})
        
        if not self.action_space.contains(action):
            self.logger.warning("アクションが行動空間外です", {
                "action": action.tolist(),
                "action_space_low": self.action_space.low.tolist(),
                "action_space_high": self.action_space.high.tolist()
            })
            # クリッピングして続行
            action = np.clip(action, self.action_space.low, self.action_space.high)
    
    def _apply_action(self, action: np.ndarray):
        """アクションの適用"""
        self.logger.log_function_entry("_apply_action", (action,))
        
        max_torque = self.robot_config.get('max_torque', 10.0)
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=max_torque
            )
        
        self.logger.debug("アクション適用完了", {
            "action": action.tolist(),
            "max_torque": max_torque
        })
    
    def _execute_physics_simulation(self):
        """物理シミュレーションの実行"""
        for step in range(self.physics_steps_per_control):
            p.stepSimulation()
            self.debug_info['physics_steps'] += 1
    
    def _get_observation(self) -> np.ndarray:
        """観測の取得"""
        try:
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
            
            # デバッグ情報の保存
            self._store_observation_debug(observation, position, euler_angles, velocity)
            
            return observation.astype(np.float32)
            
        except Exception as e:
            self.logger.error("観測取得エラー", exception=e)
            raise RobotStateError(self.robot_id) from e
    
    def _store_observation_debug(self, observation: np.ndarray, position: Tuple, 
                               euler_angles: Tuple, velocity: Tuple):
        """観測のデバッグ情報保存"""
        self.debug_info.update({
            'robot_position': position,
            'robot_orientation': euler_angles,
            'robot_velocity': velocity,
            'last_observation': observation.tolist()
        })
    
    def _calculate_reward_detailed(self, action: np.ndarray) -> Tuple[float, Dict]:
        """詳細な報酬計算"""
        reward_breakdown = {}
        
        # 1. 前進速度報酬
        velocity, _ = p.getBaseVelocity(self.robot_id)
        forward_velocity = velocity[0]
        velocity_reward = forward_velocity * self.reward_weights['forward_velocity']
        reward_breakdown['forward_velocity'] = velocity_reward
        
        # 2. 生存報酬
        survival_reward = self.reward_weights['survival']
        reward_breakdown['survival'] = survival_reward
        
        # 3. 転倒ペナルティ
        fall_penalty = 0.0
        if self._is_fallen():
            fall_penalty = self.reward_weights['fall_penalty']
        reward_breakdown['fall_penalty'] = fall_penalty
        
        # 4. エネルギー効率ペナルティ
        energy_penalty = np.sum(np.abs(action)) * abs(self.reward_weights['energy_efficiency'])
        reward_breakdown['energy_efficiency'] = -energy_penalty
        
        # 5. 高さ安定性報酬（ジャンプ抑制・安定歩行促進）
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        height = position[2]
        target_height = self.reward_weights.get('target_height', 0.1)
        height_stability = -abs(height - target_height) * self.reward_weights.get('height_stability_weight', 5.0)
        reward_breakdown['height_stability'] = height_stability
        
        # 6. 垂直速度ペナルティ（ジャンプ・落下抑制）
        vertical_velocity = velocity[2]  # Z方向の速度
        vertical_penalty = -abs(vertical_velocity) * self.reward_weights.get('vertical_velocity_penalty', 2.0)
        reward_breakdown['vertical_velocity_penalty'] = vertical_penalty
        
        # 7. 前進距離報酬（真の前進を促進）
        if not hasattr(self, 'initial_position'):
            self.initial_position = position
        distance_traveled = position[0] - self.initial_position[0]  # X方向の移動距離
        distance_reward = distance_traveled * self.reward_weights.get('distance_weight', 1.0)
        reward_breakdown['distance_traveled'] = distance_reward
        
        # 合計報酬
        total_reward = sum(reward_breakdown.values())
        reward_breakdown['total'] = total_reward
        
        # デバッグ用の詳細ログ出力
        self._log_reward_breakdown(reward_breakdown, height, vertical_velocity, distance_traveled)
        
        self.debug_info['last_reward_breakdown'] = reward_breakdown
        
        return total_reward, reward_breakdown
    
    def _log_reward_breakdown(self, reward_breakdown: Dict, height: float, vertical_velocity: float, distance_traveled: float):
        """報酬内訳の詳細ログ出力（デバッグ用）"""
        if self.episode_steps % 50 == 0:  # 50ステップごとにログ出力
            self.logger.debug("報酬内訳詳細", {
                "episode_steps": self.episode_steps,
                "height": float(height),
                "vertical_velocity": float(vertical_velocity),
                "distance_traveled": float(distance_traveled),
                "reward_breakdown": {
                    "forward_velocity": float(reward_breakdown.get('forward_velocity', 0)),
                    "survival": float(reward_breakdown.get('survival', 0)),
                    "fall_penalty": float(reward_breakdown.get('fall_penalty', 0)),
                    "energy_efficiency": float(reward_breakdown.get('energy_efficiency', 0)),
                    "height_stability": float(reward_breakdown.get('height_stability', 0)),
                    "vertical_velocity_penalty": float(reward_breakdown.get('vertical_velocity_penalty', 0)),
                    "distance_traveled": float(reward_breakdown.get('distance_traveled', 0)),
                    "total": float(reward_breakdown.get('total', 0))
                }
            })
    
    def _is_fallen(self) -> bool:
        """転倒判定"""
        try:
            # 高さチェック
            position, orientation = p.getBasePositionAndOrientation(self.robot_id)
            height = position[2]
            min_height = self.termination_config.get('min_height', 0.05)
            
            if height < min_height:
                self.logger.debug("転倒判定: 高さ不足", {"height": height, "min_height": min_height})
                return True
            
            # 姿勢チェック
            euler_angles = p.getEulerFromQuaternion(orientation)
            roll, pitch, yaw = euler_angles
            
            max_roll = np.radians(self.termination_config.get('max_roll', 45.0))
            max_pitch = np.radians(self.termination_config.get('max_pitch', 45.0))
            
            if abs(roll) > max_roll or abs(pitch) > max_pitch:
                self.logger.debug("転倒判定: 姿勢異常", {
                    "roll": np.degrees(roll),
                    "pitch": np.degrees(pitch),
                    "max_roll": np.degrees(max_roll),
                    "max_pitch": np.degrees(max_pitch)
                })
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("転倒判定エラー", exception=e)
            return True  # エラー時は安全のため転倒とみなす
    
    def _is_terminated(self) -> bool:
        """終了条件の判定"""
        return self._is_fallen()
    
    def _is_truncated(self) -> bool:
        """切り捨て条件の判定"""
        return self.episode_steps >= self.env_config['max_episode_steps']
    
    def _get_step_info(self, reward_breakdown: Dict, step_start_time: float) -> Dict:
        """ステップ情報の取得"""
        step_time = time.time() - step_start_time
        
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        velocity, _ = p.getBaseVelocity(self.robot_id)
        
        return {
            'episode_steps': self.episode_steps,
            'forward_velocity': velocity[0],
            'robot_height': position[2],
            'robot_orientation': p.getEulerFromQuaternion(orientation),
            'reward_breakdown': reward_breakdown,
            'step_time': step_time,
            'physics_steps_total': self.debug_info['physics_steps'],
            'debug_info': self.debug_info.copy()
        }
    
    def _update_internal_state(self, action: np.ndarray, reward: float, info: Dict):
        """内部状態の更新"""
        self.episode_steps += 1
        self.last_action = action.copy()
        self.debug_info['step_count'] += 1
        
        # 履歴の保存（最新1000ステップのみ）
        self.action_history.append(action.copy())
        if len(self.action_history) > 1000:
            self.action_history.pop(0)
        
        # パフォーマンス履歴
        self.performance_metrics[f"step_{self.episode_steps}"] = {
            'reward': reward,
            'step_time': info.get('step_time', 0),
            'forward_velocity': info.get('forward_velocity', 0)
        }
    
    def _get_safe_state(self) -> np.ndarray:
        """エラー時の安全な状態を返す"""
        return np.zeros(22, dtype=np.float32)
    
    def _cleanup_on_error(self):
        """エラー時のクリーンアップ"""
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
    
    def render(self) -> Optional[np.ndarray]:
        """環境の可視化"""
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        return None
    
    def close(self):
        """環境の終了"""
        self.logger.info("環境のクリーンアップ開始")
        
        try:
            if self.physics_client is not None:
                p.disconnect(self.physics_client)
                self.physics_client = None
            
            self.logger.info("環境のクリーンアップ完了")
            
        except Exception as e:
            self.logger.error("環境クリーンアップ中にエラーが発生しました", exception=e)
    
    def get_debug_info(self) -> Dict:
        """デバッグ情報の取得"""
        return {
            'debug_info': self.debug_info.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'config': self.config.copy(),
            'joint_info': self.joint_info.copy() if hasattr(self, 'joint_info') else {},
            'recent_actions': self.action_history[-10:] if self.action_history else [],
            'episode_stats': {
                'total_episodes': self.total_episodes,
                'current_episode_steps': self.episode_steps,
                'physics_steps_total': self.debug_info.get('physics_steps', 0)
            }
        }
