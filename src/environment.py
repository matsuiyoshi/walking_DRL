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
            'distance_weight': self.reward_config.get('distance_weight', 1.0),
            
            # 足先接地報酬（新規追加）
            'foot_contact_reward': self.reward_config.get('foot_contact_reward', 0.0),
            'proper_gait_reward': self.reward_config.get('proper_gait_reward', 0.0),
            'body_clearance_reward': self.reward_config.get('body_clearance_reward', 0.0),
            'height_reward_weight': self.reward_config.get('height_reward_weight', 0.0)
        }
        
        # 足先リンクの特定
        self._identify_foot_links()
        
        self.logger.info("報酬関数設定完了", {
            "reward_weights": self.reward_weights,
            "velocity_proportional_foot_reward": True
        })
    
    def _identify_foot_links(self):
        """
        足先リンク（エンドエフェクタ）とshoulder-linkを特定
        
        Bittleの構造:
        - 各脚に2つの関節（shoulder, knee）
        - kneeリンクが足先（エンドエフェクタ）
        - shoulderリンクも衝突形状を持つため、接地を防ぐ必要がある
        """
        self.foot_links = []
        self.shoulder_links = []  # 新規追加: shoulder-linkを別カテゴリとして管理
        self.body_links = []  # 胴体と中間関節
        
        # 足先リンク名のパターン（kneeが足先）
        foot_link_patterns = ['knee']
        
        # shoulderリンク名のパターン（新規追加）
        shoulder_link_patterns = ['shoulder']
        
        # 胴体リンク名のパターン（shoulderを除外）
        body_link_patterns = ['base', 'battery', 'cover', 'mainboard', 'imu']
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8').lower()  # リンク名
            
            # 足先リンクの判定
            if any(pattern in link_name for pattern in foot_link_patterns):
                self.foot_links.append(i)
                self.logger.debug(f"足先リンク検出: {link_name} (link_index: {i})")
            
            # shoulderリンクの判定（新規追加）
            elif any(pattern in link_name for pattern in shoulder_link_patterns):
                self.shoulder_links.append(i)
                self.logger.debug(f"Shoulderリンク検出: {link_name} (link_index: {i})")
            
            # 胴体・中間関節リンクの判定
            elif any(pattern in link_name for pattern in body_link_patterns):
                self.body_links.append(i)
                self.logger.debug(f"胴体リンク検出: {link_name} (link_index: {i})")
        
        # base_link（リンクインデックス-1）も胴体として追加
        self.body_links.append(-1)
        
        self.logger.info(f"リンク検出完了: 足先={len(self.foot_links)}個, shoulder={len(self.shoulder_links)}個, 胴体={len(self.body_links)}個", {
            "foot_links": self.foot_links,
            "shoulder_links": self.shoulder_links,
            "body_links_count": len(self.body_links)
        })
        
        # 検証: 4本の足が検出されているか
        if len(self.foot_links) != 4:
            self.logger.warning(f"期待される足先リンク数は4個ですが、{len(self.foot_links)}個検出されました")
        
        # 検証: 4つのshoulderが検出されているか
        if len(self.shoulder_links) != 4:
            self.logger.warning(f"期待されるshoulderリンク数は4個ですが、{len(self.shoulder_links)}個検出されました")
        
        # 運動学的パラメータの初期化
        self._initialize_kinematic_parameters()
    
    def _initialize_kinematic_parameters(self):
        """
        運動学的計算のためのパラメータを初期化
        Bittleの脚の構造パラメータ（実測値）
        """
        # リンク長さ（メートル単位、Petoi公式/実測値）
        self.shoulder_length = 0.0  # 肩関節の長さ（肩の回転軸）
        self.upper_leg_length = 0.06  # 肩から膝までの距離（6cm）
        self.lower_leg_length = 0.06  # 膝から足先までの距離（6cm）
        
        # 各脚の肩関節のベース座標系での位置（URDFから取得）
        # [x, y, z] の順序
        self.leg_base_positions = {
            'left_front': [-0.44596, 0.52264, -0.02102],
            'right_front': [0.45149, 0.52264, -0.02102],
            'left_back': [-0.44596, -0.51923, -0.02102],
            'right_back': [0.45149, -0.51923, -0.02102]
        }
        
        # 関節インデックスと脚の対応付け
        # Bittleの関節順序に基づく
        self.leg_joint_mapping = {
            0: 'left_front',   # left-front-shoulder
            1: 'left_front',   # left-front-knee
            2: 'right_front',  # right-front-shoulder
            3: 'right_front',  # right-front-knee
            4: 'left_back',    # left-back-shoulder
            5: 'left_back',    # left-back-knee
            6: 'right_back',   # right-back-shoulder
            7: 'right_back'    # right-back-knee
        }
        
        # 左右対称を考慮するための符号（右側は時計回り反転）
        self.leg_symmetry_sign = {
            'left_front': 1.0,
            'right_front': -1.0,
            'left_back': 1.0,
            'right_back': -1.0
        }
        
        self.logger.info("運動学的パラメータ初期化完了", {
            "upper_leg_length": self.upper_leg_length,
            "lower_leg_length": self.lower_leg_length,
            "leg_base_positions": self.leg_base_positions
        })
    
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
        
        # 地面の追加（シンプルな地面を直接作成）
        plane_shape = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, plane_shape)
        self.logger.debug("シンプルな地面を作成しました")
        
        self.logger.debug("物理シミュレーションリセット完了")
    
    def _reset_robot_state(self):
        """ロボット状態のリセット"""
        self.logger.debug("ロボット状態リセット開始")
        
        try:
            # ロボットの再読み込み
            self._load_robot()
            
            # 初期関節角度の設定（±10度のランダム初期化）
            initial_joint_angles = self._generate_random_joint_angles()
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
    
    def _generate_random_joint_angles(self) -> List[float]:
        """
        足先で立つ初期姿勢を生成（±10度のランダム性付き）
        
        Bittleの足先で立つための適切な初期姿勢：
        - Shoulder関節: 約45度前方（+0.785 rad）
        - Knee関節: 約-70度（-1.2 rad）→足先が下向き
        
        Returns:
            List[float]: 8つの関節角度（shoulder, knee交互に4脚分）
        """
        # 足先で立つための基本姿勢
        base_shoulder_angle = 0.785  # 45度（前方）
        base_knee_angle = -1.2       # -70度（下向き、膝を曲げる）
        
        # ±10度のランダム性を追加（探索のため）
        max_deviation = np.radians(10.0)  # 10度
        
        random_angles = []
        for i in range(len(self.joint_indices)):
            # 偶数インデックス: shoulder関節（0, 2, 4, 6）
            # 奇数インデックス: knee関節（1, 3, 5, 7）
            if i % 2 == 0:
                # Shoulder関節
                base_angle = base_shoulder_angle
            else:
                # Knee関節
                base_angle = base_knee_angle
            
            # ランダム性を追加
            angle = base_angle + np.random.uniform(-max_deviation, max_deviation)
            random_angles.append(float(angle))
        
        self.logger.debug("足先立ち初期姿勢生成", {
            "base_shoulder_deg": float(np.degrees(base_shoulder_angle)),
            "base_knee_deg": float(np.degrees(base_knee_angle)),
            "angles_degrees": [float(np.degrees(angle)) for angle in random_angles],
            "angles_radians": [float(angle) for angle in random_angles]
        })
        
        return random_angles
    
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
        
        # デバッグ: ステップ開始ログ
        self.logger.debug("ステップ開始", {
            "episode_steps": self.episode_steps,
            "action": action.tolist()[:4]  # 最初の4要素のみ表示
        })
        
        try:
            # アクションの検証
            self._validate_action(action)
            self.logger.debug("アクション検証完了")
            
            # アクションの適用
            self._apply_action(action)
            self.logger.debug("アクション適用完了")
            
            # 物理シミュレーションの実行
            self._execute_physics_simulation()
            self.logger.debug("物理シミュレーション完了")
            
            # 状態の取得
            observation = self._get_observation()
            self.logger.debug("観測取得完了")
            
            # 報酬の計算
            reward, reward_breakdown = self._calculate_reward_detailed(action)
            self.logger.debug("報酬計算完了", {"reward": float(reward)})
            
            # 終了条件の判定
            terminated = self._is_terminated()
            truncated = self._is_truncated()
            self.logger.debug("終了条件判定完了", {
                "terminated": terminated,
                "truncated": truncated,
                "episode_steps": self.episode_steps
            })
            
            # メタ情報の更新
            info = self._get_step_info(reward_breakdown, step_start_time)
            
            # 内部状態の更新前のログ
            self.logger.debug("内部状態更新前", {
                "episode_steps_before": self.episode_steps
            })
            
            # 内部状態の更新
            self._update_internal_state(action, reward, info)
            
            # 内部状態の更新後のログ
            self.logger.debug("内部状態更新後", {
                "episode_steps_after": self.episode_steps
            })
            
            # ステップ完了ログ
            self.logger.debug("ステップ完了", {
                "episode_steps": self.episode_steps,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated
            })
            
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
            
            # 個別クリッピングの適用
            observation = self._apply_observation_clipping(observation)
            
            # デバッグ情報の保存
            self._store_observation_debug(observation, position, euler_angles, velocity)
            
            return observation.astype(np.float32)
            
        except Exception as e:
            self.logger.error("観測取得エラー", exception=e)
            raise RobotStateError(self.robot_id) from e
    
    def _apply_observation_clipping(self, observation: np.ndarray) -> np.ndarray:
        """観測値の個別クリッピング（Bittleサイズ最適化）"""
        clipped_obs = observation.copy()
        
        # 関節角度 (0-7): [-π, π] → [-3.2, 3.2] (ラジアン)
        clipped_obs[0:8] = np.clip(clipped_obs[0:8], -3.2, 3.2)
        
        # 姿勢角度 (8-10): [-π, π] → [-3.2, 3.2] (ラジアン)
        clipped_obs[8:11] = np.clip(clipped_obs[8:11], -3.2, 3.2)
        
        # 速度 (11-13): Bittleサイズに適した範囲 → [-2.0, 2.0] (m/s)
        clipped_obs[11:14] = np.clip(clipped_obs[11:14], -2.0, 2.0)
        
        # 前回アクション (14-21): [-π, π] → [-3.2, 3.2] (ラジアン)
        clipped_obs[14:22] = np.clip(clipped_obs[14:22], -3.2, 3.2)
        
        return clipped_obs
    
    def _store_observation_debug(self, observation: np.ndarray, position: Tuple, 
                               euler_angles: Tuple, velocity: Tuple):
        """観測のデバッグ情報保存"""
        self.debug_info.update({
            'robot_position': position,
            'robot_orientation': euler_angles,
            'robot_velocity': velocity,
            'last_observation': observation.tolist()
        })
    
    def _calculate_foot_contact_reward(self) -> Tuple[float, Dict]:
        """
        足先接地報酬の計算（速度比例型） + shoulder/knee接地ペナルティ
        
        Returns:
            Tuple[float, Dict]: (合計報酬, 報酬詳細)
        """
        foot_contact_count = 0
        body_contact = False
        knee_contact = False
        shoulder_contact = False  # 新規追加: shoulder接地フラグ
        foot_contact_details = {}
        reward_breakdown = {}
        
        # 前進速度を取得（速度比例報酬のため）
        velocity, _ = p.getBaseVelocity(self.robot_id)
        forward_velocity = max(velocity[0], 0.0)  # 前進のみカウント（後退は報酬なし）
        
        # 膝リンクのインデックスを特定
        knee_links = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8').lower()
            if 'knee' in link_name:
                knee_links.append(i)
        
        # 全ての接触点を取得
        contact_points = p.getContactPoints(self.robot_id)
        
        for contact in contact_points:
            link_index = contact[3]  # ロボット側のリンクインデックス
            
            # 足先との接触
            if link_index in self.foot_links:
                foot_contact_count += 1
                foot_contact_details[f'foot_{link_index}'] = True
            
            # shoulderリンクとの接触（新規追加：非常に大きなペナルティ）
            elif link_index in self.shoulder_links:
                shoulder_contact = True
                self.logger.debug(f"Shoulder接触検出（物理エンジン）: link_index={link_index}")
            
            # 膝リンクとの接触（ペナルティ対象）
            elif link_index in knee_links:
                knee_contact = True
                self.logger.debug(f"膝接触検出（物理エンジン）: link_index={link_index}")
            
            # 胴体や中間関節との接触（ペナルティ対象）
            elif link_index in self.body_links:
                body_contact = True
        
        # 1. 足先接地報酬（速度比例型：足先接地しながら速く動くことを促進）
        if self.reward_weights['foot_contact_reward'] != 0:
            if 1 <= foot_contact_count <= 4:
                # 速度比例型: 接地率 × 前進速度 × 係数
                foot_reward = self.reward_weights['foot_contact_reward'] * (foot_contact_count / 4.0) * forward_velocity
            else:
                foot_reward = 0.0
            reward_breakdown['foot_contact'] = foot_reward
        
        # 2. 適切な歩容報酬（2-3本で歩行が理想的）
        if self.reward_weights['proper_gait_reward'] != 0:
            if 2 <= foot_contact_count <= 3:
                gait_reward = self.reward_weights['proper_gait_reward']
            else:
                gait_reward = 0.0
            reward_breakdown['proper_gait'] = gait_reward
        
        # 3. 胴体クリアランス報酬（胴体が地面に接触していないこと）
        if self.reward_weights['body_clearance_reward'] != 0:
            if not body_contact:
                clearance_reward = self.reward_weights['body_clearance_reward']
            else:
                clearance_reward = -self.reward_weights['body_clearance_reward']  # ペナルティ
            reward_breakdown['body_clearance'] = clearance_reward
        
        # 4. 胴体高さ報酬（足先で立つことを促進）
        if self.reward_weights['height_reward_weight'] != 0:
            position, _ = p.getBasePositionAndOrientation(self.robot_id)
            current_height = position[2]
            target_height = self.reward_weights.get('target_height', 0.12)
            
            # 目標高さに近いほど高報酬
            height_error = abs(current_height - target_height)
            if height_error < 0.03:  # 3cm以内
                height_reward = self.reward_weights['height_reward_weight'] * (1.0 - height_error / 0.03)
            else:
                # 低すぎる場合はペナルティ
                height_reward = -self.reward_weights['height_reward_weight'] * min(height_error, 0.1)
            reward_breakdown['height_maintenance'] = height_reward
        
        # 5. 膝接地ペナルティ（膝で這うことを防ぐ）
        if knee_contact:
            knee_penalty = self.reward_config.get('knee_contact_penalty', -200.0)
            reward_breakdown['knee_contact_penalty'] = knee_penalty
            self.logger.debug("膝接地ペナルティ適用（物理エンジン）", {"penalty": knee_penalty})
        
        # 6. Shoulder接地ペナルティ（新規追加：非常に大きなペナルティ）
        if shoulder_contact:
            shoulder_penalty = self.reward_config.get('shoulder_contact_penalty', -500.0)
            reward_breakdown['shoulder_contact_penalty'] = shoulder_penalty
            self.logger.warning("Shoulder接地ペナルティ適用（物理エンジン）", {"penalty": shoulder_penalty})
        
        # デバッグ情報を保存
        if hasattr(self, 'debug_info'):
            self.debug_info['last_foot_contact'] = {
                'foot_contact_count': foot_contact_count,
                'body_contact': body_contact,
                'knee_contact': knee_contact,
                'shoulder_contact': shoulder_contact,  # 新規追加
                'foot_details': foot_contact_details
            }
        
        total_reward = sum(reward_breakdown.values())
        return total_reward, reward_breakdown
    
    def _calculate_foot_positions_from_kinematics(self) -> Dict[str, Tuple[float, float, float]]:
        """
        関節角度から三角関数を使って足先位置を計算（運動学的計算）
        2リンクマニピュレータの順運動学（X-Z平面）
        
        Returns:
            Dict[str, Tuple[float, float, float]]: 各脚の足先位置（ワールド座標）
                キー: 'left_front', 'right_front', 'left_back', 'right_back'
                値: (x, y, z) タプル
        """
        foot_positions = {}
        
        # ロボットのベース位置と姿勢を取得
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        
        # 関節角度を取得
        joint_angles = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_angles.append(joint_state[0])
        
        # 各脚の足先位置を計算
        legs = ['left_front', 'right_front', 'left_back', 'right_back']
        
        for leg_idx, leg_name in enumerate(legs):
            # 肩関節と膝関節の角度を取得
            shoulder_joint_idx = leg_idx * 2
            knee_joint_idx = leg_idx * 2 + 1
            
            shoulder_angle = joint_angles[shoulder_joint_idx]
            knee_angle = joint_angles[knee_joint_idx]
            
            # 左右対称の考慮（モーターの回転方向）
            symmetry_sign = self.leg_symmetry_sign[leg_name]
            shoulder_angle_adj = shoulder_angle * symmetry_sign
            knee_angle_adj = knee_angle * symmetry_sign
            
            # 肩関節のベース座標系での位置
            leg_base_pos = self.leg_base_positions[leg_name]
            
            # === 2リンクマニピュレータの順運動学（X-Z平面） ===
            L1 = self.upper_leg_length  # shoulder → knee（6cm）
            L2 = self.lower_leg_length  # knee → foot tip（6cm）
            
            # 第1リンク（shoulder → knee）の末端位置
            # X-Z平面での回転（Y軸まわり）
            knee_x_local = L1 * np.sin(shoulder_angle_adj)
            knee_z_local = -L1 * np.cos(shoulder_angle_adj)  # 下向きが負（重力方向）
            
            # 第2リンク（knee → foot tip）
            # 膝関節の絶対角度 = shoulder角度 + knee角度
            foot_angle_abs = shoulder_angle_adj + knee_angle_adj
            foot_x_from_knee = L2 * np.sin(foot_angle_abs)
            foot_z_from_knee = -L2 * np.cos(foot_angle_abs)
            
            # 足先のローカル座標（肩関節基準、X-Z平面）
            foot_x_local = knee_x_local + foot_x_from_knee
            foot_z_local = knee_z_local + foot_z_from_knee
            
            # ベース座標系での足先位置
            foot_local_x = leg_base_pos[0] + foot_x_local
            foot_local_y = leg_base_pos[1]
            foot_local_z = leg_base_pos[2] + foot_z_local
            
            # ローカル座標からワールド座標への変換
            foot_world_pos = self._transform_local_to_world(
                [foot_local_x, foot_local_y, foot_local_z],
                base_position,
                base_orientation
            )
            
            foot_positions[leg_name] = foot_world_pos
            
            # デバッグログ（詳細な計算過程を100ステップごとに出力）
            if self.episode_steps % 100 == 0 and leg_idx == 0:  # 左前脚のみ
                self.logger.debug(f"{leg_name} 足先位置計算（X-Z平面）", {
                    "shoulder_angle_deg": float(np.degrees(shoulder_angle)),
                    "knee_angle_deg": float(np.degrees(knee_angle)),
                    "knee_local": (float(knee_x_local), float(knee_z_local)),
                    "foot_local": (float(foot_x_local), float(foot_z_local)),
                    "foot_world_z": float(foot_world_pos[2])
                })
        
        return foot_positions
    
    def _transform_local_to_world(self, local_pos: List[float],
                                  base_pos: Tuple[float, float, float],
                                  base_quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """
        ローカル座標からワールド座標への変換
        
        Args:
            local_pos: ローカル座標系での位置 [x, y, z]
            base_pos: ベースのワールド座標系での位置
            base_quat: ベースの姿勢（クォータニオン）
        
        Returns:
            ワールド座標系での位置 (x, y, z)
        """
        # クォータニオンを回転行列に変換
        rotation_matrix = p.getMatrixFromQuaternion(base_quat)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        
        # ローカル位置を回転
        local_array = np.array(local_pos)
        rotated_pos = rotation_matrix @ local_array
        
        # ベース位置を加算
        world_pos = rotated_pos + np.array(base_pos)
        
        return tuple(world_pos)
    
    def _calculate_foot_ground_distances(self) -> Dict[str, float]:
        """
        足先と地面との距離を計算（運動学的計算）
        
        Returns:
            Dict[str, float]: 各脚の地面からの距離
                キー: 'left_front', 'right_front', 'left_back', 'right_back'
                値: 地面からの距離（メートル）
        """
        foot_positions = self._calculate_foot_positions_from_kinematics()
        ground_distances = {}
        
        for leg_name, foot_pos in foot_positions.items():
            # 地面は z=0 にあると仮定
            ground_distance = foot_pos[2]
            ground_distances[leg_name] = ground_distance
        
        return ground_distances
    
    def _calculate_foot_contact_reward_kinematic(self) -> Tuple[float, Dict]:
        """
        運動学的計算による足先接地報酬の計算（速度比例型） + shoulder高さチェック
        
        Returns:
            Tuple[float, Dict]: (合計報酬, 報酬詳細)
        """
        reward_breakdown = {}
        
        # 足先と地面との距離を運動学的に計算
        ground_distances = self._calculate_foot_ground_distances()
        
        # 接地判定の閾値（3cm以内なら接地とみなす）
        contact_threshold = self.reward_config.get('contact_threshold', 0.03)
        
        # 前進速度を取得（速度比例報酬のため）
        velocity, _ = p.getBaseVelocity(self.robot_id)
        forward_velocity = max(velocity[0], 0.0)  # 前進のみカウント（後退は報酬なし）
        
        # 接地している足をカウント
        foot_contact_count = 0
        foot_contact_details = {}
        
        for leg_name, distance in ground_distances.items():
            if distance <= contact_threshold:
                foot_contact_count += 1
                foot_contact_details[leg_name] = True
            else:
                foot_contact_details[leg_name] = False
        
        # 1. 足先接地報酬（速度比例型：足先接地しながら速く動くことを促進）
        if self.reward_weights.get('foot_contact_reward', 0.0) != 0:
            if 1 <= foot_contact_count <= 4:
                # 速度比例型: 接地率 × 前進速度 × 係数
                foot_reward = self.reward_weights['foot_contact_reward'] * (foot_contact_count / 4.0) * forward_velocity
            else:
                foot_reward = 0.0
            reward_breakdown['foot_contact'] = foot_reward
        
        # 2. 適切な歩容報酬（2-3本で歩行が理想的）
        if self.reward_weights.get('proper_gait_reward', 0.0) != 0:
            if 2 <= foot_contact_count <= 3:
                gait_reward = self.reward_weights['proper_gait_reward']
            else:
                gait_reward = 0.0
            reward_breakdown['proper_gait'] = gait_reward
        
        # 3. 胴体高さ維持報酬（高さが低すぎる場合は大きなペナルティ）
        if self.reward_weights.get('height_reward_weight', 0.0) != 0:
            position, _ = p.getBasePositionAndOrientation(self.robot_id)
            current_height = position[2]
            target_height = self.reward_config.get('target_height', 0.12)
            
            height_error = abs(current_height - target_height)
            if height_error < 0.03:  # 3cm以内
                height_reward = self.reward_weights['height_reward_weight'] * (1.0 - height_error / 0.03)
            else:
                # 低すぎる場合は非常に大きなペナルティ（這う動作を防ぐ）
                if current_height < 0.08:  # 8cm未満なら這っている
                    height_reward = -self.reward_weights['height_reward_weight'] * 10.0
                    self.logger.debug("胴体が低すぎる（這い動作）", {
                        "current_height": float(current_height),
                        "penalty": float(height_reward)
                    })
                else:
                    height_reward = -self.reward_weights['height_reward_weight'] * min(height_error, 0.1)
            reward_breakdown['height_maintenance'] = height_reward
        
        # 4. Shoulder高さチェック（新規追加：運動学的にshoulder位置を計算）
        shoulder_too_low = self._check_shoulder_height_kinematic()
        if shoulder_too_low:
            shoulder_penalty = self.reward_config.get('shoulder_contact_penalty', -500.0)
            reward_breakdown['shoulder_height_penalty'] = shoulder_penalty
            self.logger.warning("Shoulder高さペナルティ適用（運動学的）", {"penalty": shoulder_penalty})
        
        # デバッグ情報の保存
        if hasattr(self, 'debug_info'):
            self.debug_info['kinematic_foot_contact'] = {
                'ground_distances': ground_distances,
                'foot_contact_count': foot_contact_count,
                'foot_details': foot_contact_details,
                'contact_threshold': contact_threshold,
                'shoulder_too_low': shoulder_too_low  # 新規追加
            }
        
        total_reward = sum(reward_breakdown.values())
        return total_reward, reward_breakdown
    
    def _check_shoulder_height_kinematic(self) -> bool:
        """
        運動学的にshoulder高さをチェック（地面に近すぎる場合はTrue）
        
        Returns:
            bool: shoulderが地面に近すぎる場合はTrue（2本以上のshoulderが閾値未満）
        """
        # ロボットのベース位置と姿勢を取得
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        
        # 関節角度を取得
        joint_angles = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_angles.append(joint_state[0])
        
        shoulder_too_low_count = 0
        shoulder_height_threshold = 0.04  # 4cm未満ならshoulder接地とみなす
        
        legs = ['left_front', 'right_front', 'left_back', 'right_back']
        
        for leg_idx, leg_name in enumerate(legs):
            shoulder_joint_idx = leg_idx * 2
            shoulder_angle = joint_angles[shoulder_joint_idx]
            
            # 左右対称の考慮（モーターの回転方向）
            symmetry_sign = self.leg_symmetry_sign[leg_name]
            shoulder_angle_adj = shoulder_angle * symmetry_sign
            
            # shoulderのローカル位置（肩関節の基準位置）
            leg_base_pos = self.leg_base_positions[leg_name]
            
            # shoulderリンクの高さを計算（簡略化：ベース高さ + オフセット）
            # shoulder関節の回転を考慮（簡略化のため、Z軸方向の変位のみ）
            shoulder_z_offset = 0.0  # shoulder関節は回転のみのため、大きな変位はない
            shoulder_z_local = leg_base_pos[2] + shoulder_z_offset
            
            # ワールド座標でのshoulder高さ
            shoulder_world_z = base_position[2] + shoulder_z_local
            
            # デバッグ（100ステップごと）
            if self.episode_steps % 100 == 0 and leg_idx == 0:
                self.logger.debug(f"{leg_name} shoulder高さチェック", {
                    "shoulder_world_z": float(shoulder_world_z),
                    "threshold": shoulder_height_threshold,
                    "base_height": float(base_position[2])
                })
            
            # 地面からの距離をチェック
            if shoulder_world_z < shoulder_height_threshold:
                shoulder_too_low_count += 1
        
        # 2本以上のshoulderが地面に近い場合はペナルティ
        return shoulder_too_low_count >= 2
    
    def _calculate_reward_detailed(self, action: np.ndarray) -> Tuple[float, Dict]:
        """VecNormalize対応の報酬計算（正規化なし）"""
        reward_breakdown = {}
        
        # 物理状態の取得
        velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        euler_angles = p.getEulerFromQuaternion(orientation)
        
        # 1. 前進速度報酬（正規化なし、生の値を使用）
        forward_velocity = velocity[0]
        velocity_reward = forward_velocity * self.reward_weights['forward_velocity']
        reward_breakdown['forward_velocity'] = velocity_reward
        
        # 2. 生存報酬（固定値）
        survival_reward = self.reward_weights['survival']
        reward_breakdown['survival'] = survival_reward
        
        # 3. 転倒ペナルティ（有効化）
        if self._is_fallen():
            fall_penalty = self.reward_weights['fall_penalty']
        else:
            fall_penalty = 0.0
        reward_breakdown['fall_penalty'] = fall_penalty
        
        # 4. エネルギー効率ペナルティ（生の角度変化量）
        if hasattr(self, 'last_action') and self.last_action is not None:
            action_change = np.abs(action - self.last_action)
            total_change = np.sum(action_change)
            energy_penalty = total_change * abs(self.reward_weights['energy_efficiency'])
        else:
            energy_penalty = 0.0
        reward_breakdown['energy_efficiency'] = -energy_penalty
        
        # 5. 姿勢安定性報酬（生の角度誤差）
        roll, pitch, yaw = euler_angles
        roll_error = abs(roll)
        pitch_error = abs(pitch)
        orientation_error = (roll_error + pitch_error) / 2
        
        orientation_penalty = -orientation_error * self.reward_weights.get('orientation_stability_weight', 0.5)
        reward_breakdown['orientation_stability'] = orientation_penalty
        
        # 6. 足先接地報酬（新規追加）
        # 運動学的計算を使用するか、物理エンジンの接触検知を使用するかを選択
        use_kinematic_contact = self.reward_config.get('use_kinematic_contact', False)
        
        if hasattr(self, 'foot_links'):
            if use_kinematic_contact:
                # 運動学的計算による足先接地報酬
                foot_reward, foot_breakdown = self._calculate_foot_contact_reward_kinematic()
            else:
                # 物理エンジンの接触検知による足先接地報酬（既存）
                foot_reward, foot_breakdown = self._calculate_foot_contact_reward()
            reward_breakdown.update(foot_breakdown)
        
        # 合計報酬の計算
        total_reward = sum(reward_breakdown.values())
        reward_breakdown['total'] = total_reward
        
        # デバッグ情報の更新
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
            # 姿勢チェックのみ
            position, orientation = p.getBasePositionAndOrientation(self.robot_id)
            euler_angles = p.getEulerFromQuaternion(orientation)
            roll, pitch, yaw = euler_angles
            
            max_roll = np.radians(self.termination_config.get('max_roll', 45.0))
            max_pitch = np.radians(self.termination_config.get('max_pitch', 45.0))
            
            if abs(roll) > max_roll or abs(pitch) > max_pitch:
                self.logger.info("転倒判定: 姿勢異常", {
                    "roll_degrees": float(np.degrees(roll)),
                    "pitch_degrees": float(np.degrees(pitch)),
                    "max_roll_degrees": float(np.degrees(max_roll)),
                    "max_pitch_degrees": float(np.degrees(max_pitch)),
                    "episode_steps": self.episode_steps,
                    "robot_position": [float(x) for x in position]
                })
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("転倒判定エラー", exception=e)
            return True  # エラー時は安全のため転倒とみなす
    
    def _is_terminated(self) -> bool:
        """終了条件の判定（転倒判定を有効化）"""
        # 転倒判定を有効化
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
