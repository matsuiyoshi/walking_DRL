"""
Bittle四足歩行ロボットの学習スクリプト
Stable-Baselines3を使用した深層強化学習の実装
"""

import os
import time
import glob
import shutil
from typing import Dict, List, Optional, Any
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import cv2

from .environment import BittleEnvironment
from .utils.logger import get_logger
from .utils.config_validator import load_and_validate_config
from .utils.exceptions import ModelLoadError


class DebugCallback(BaseCallback):
    """デバッグ用コールバック"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # エピソード終了時の処理
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if info.get('episode'):
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    self.logger.info("エピソード完了", {
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
                        "mean_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
                    })
        
        return True
    
    def _on_training_end(self) -> None:
        """学習終了時の処理"""
        self.logger.info("学習統計", {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "std_reward": float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
            "min_reward": float(np.min(self.episode_rewards)) if self.episode_rewards else 0.0,
            "max_reward": float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0
        })


class VideoEvalCallback(EvalCallback):
    """動画保存機能付き評価コールバック（修正版）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_logger = get_logger("video_eval_callback")
        self.video_dir = os.path.join(self.log_path, 'evaluation_videos')
        os.makedirs(self.video_dir, exist_ok=True)
        self.frames = []
        self.eval_count = 0
        self.is_evaluating = False  # 評価状態フラグを追加
        self.robot_id = None  # ロボットIDを保存
        self.physics_client = None  # PhysicsClientを保存
        
    def on_eval_start(self):
        """評価開始時の処理"""
        super().on_eval_start()
        
        # 評価状態フラグをTrueに設定
        self.is_evaluating = True
        
        # 評価回数のカウント
        self.eval_count += 1
        
        # ロボット情報の取得
        self._get_robot_info()
        
        # 動画保存用のディレクトリ作成（修正版）
        eval_timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.current_video_dir = os.path.join(
            self.log_path, 'evaluation_videos', 
            f'eval_{self.eval_count:03d}_{eval_timestamp}'
        )
        
        try:
            os.makedirs(self.current_video_dir, exist_ok=True)
            self.video_logger.info(f"動画ディレクトリ作成完了: {self.current_video_dir}")
        except Exception as dir_e:
            self.video_logger.error(f"動画ディレクトリ作成エラー: {dir_e}")
            # フォールバック: 親ディレクトリを使用
            self.current_video_dir = os.path.join(self.log_path, 'evaluation_videos')
            os.makedirs(self.current_video_dir, exist_ok=True)
        
        # フレームの初期化
        self.frames = []
        
        self.video_logger.info(f"評価動画保存開始 (評価回数: {self.eval_count})", {
            "video_dir": self.current_video_dir,
            "robot_id": self.robot_id,
            "physics_client": self.physics_client
        })
    
    def on_eval_end(self):
        """評価終了時の処理"""
        super().on_eval_end()
        
        # 評価状態フラグをFalseに設定
        self.is_evaluating = False
        
        # 動画の保存
        self.save_evaluation_video()
        
        self.video_logger.info(f"評価動画保存終了 (評価回数: {self.eval_count})", {
            "frame_count": len(self.frames)
        })
    
    def _get_robot_info(self):
        """ロボット情報の取得"""
        try:
            # 環境からロボット情報を取得
            if hasattr(self.eval_env, 'envs') and len(self.eval_env.envs) > 0:
                base_env = self.eval_env.envs[0]
                
                # VecNormalizeでラップされている場合の正しい構造
                if hasattr(base_env, 'env'):
                    actual_env = base_env.env
                else:
                    actual_env = base_env
                
                # BittleEnvironmentからロボット情報を取得
                if hasattr(actual_env, 'robot_id') and hasattr(actual_env, 'physics_client'):
                    self.robot_id = actual_env.robot_id
                    self.physics_client = actual_env.physics_client
                    
                    self.video_logger.debug("ロボット情報取得完了", {
                        "robot_id": self.robot_id,
                        "physics_client": self.physics_client
                    })
                else:
                    self.video_logger.warning("ロボット情報を取得できませんでした")
            else:
                self.video_logger.warning("評価環境からロボット情報を取得できませんでした")
                
        except Exception as e:
            self.video_logger.error("ロボット情報取得エラー", exception=e)
    
    def _get_robot_position(self):
        """ロボットの現在位置を取得"""
        try:
            import pybullet as p
            
            if self.robot_id is None or self.physics_client is None:
                return [0, 0, 0.15]  # デフォルト位置
            
            position, _ = p.getBasePositionAndOrientation(
                self.robot_id, 
                physicsClientId=self.physics_client
            )
            return position
            
        except Exception as e:
            self.video_logger.debug(f"ロボット位置取得エラー: {e}")
            return [0, 0, 0.15]  # デフォルト位置


    def _get_pybullet_frame(self):
        """PyBullet環境からフレームを取得（修正版）"""
        try:
            import pybullet as p
            
            if self.physics_client is None:
                self.video_logger.debug("PhysicsClientが設定されていません")
                return None
            
            # カメラパラメータの設定
            width, height = 640, 480
            
            # ロボットの現在位置を取得
            robot_position = self._get_robot_position()
            
            # PhysicsClientの接続確認
            try:
                current_client = p.getConnectionInfo(physicsClientId=self.physics_client)
                if current_client['isConnected'] == 0:
                    self.video_logger.debug("PhysicsClientが接続されていません")
                    return None
            except Exception as e:
                self.video_logger.debug(f"PhysicsClient接続確認エラー: {e}")
                return None
            
            # ビューマトリックスの計算（ロボットを動的に追跡）
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=robot_position,  # ✅ ロボットの実際の位置
                distance=1.5,  # カメラ距離
                yaw=45,  # 斜め45度から見る
                pitch=-30,  # 少し上から見下ろす
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client  # ✅ 修正: physicsClientIdを指定
            )
            
            # プロジェクションマトリックスの計算
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60,  # 視野角
                aspect=width/height,
                nearVal=0.1,
                farVal=100,
                physicsClientId=self.physics_client  # ✅ 修正: physicsClientIdを指定
            )
            
            # カメラ画像の取得
            _, _, rgb_array, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.physics_client  # ✅ 修正: physicsClientIdを指定
            )
            
            # RGB配列を適切な形状に変換
            rgb_array = np.array(rgb_array).reshape(height, width, 4)[:, :, :3]  # RGBA → RGB
            
            return rgb_array.astype(np.uint8)
            
        except Exception as e:
            self.video_logger.debug(f"PyBulletフレーム取得エラー: {e}")
            return None
    
    def save_evaluation_video(self):
        """評価動画の保存"""
        if not self.frames:
            self.video_logger.warning("保存するフレームがありません")
            return
        
        # 動画ファイル名の生成
        eval_info = self._get_evaluation_info()
        video_filename = self._generate_video_filename(eval_info)
        video_path = os.path.join(self.current_video_dir, video_filename)
        
        try:
            # フレームを動画に保存
            self._save_frames_as_video(self.frames, video_path)
            
            self.video_logger.info("評価動画保存完了", {
                "video_path": video_path,
                "frame_count": len(self.frames),
                "eval_info": eval_info
            })
            
        except Exception as e:
            self.video_logger.error("動画保存エラー", exception=e)
    
    def _get_evaluation_info(self):
        """評価情報の取得"""
        return {
            "eval_count": self.eval_count,
            "timestamp": time.strftime('%Y%m%d_%H%M%S'),
            "frame_count": len(self.frames)
        }
    
    def _generate_video_filename(self, eval_info):
        """動画ファイル名の生成"""
        eval_count = eval_info['eval_count']
        timestamp = eval_info['timestamp']
        frame_count = eval_info['frame_count']
        
        filename = f"evaluation_video_{eval_count:03d}_{timestamp}_frames_{frame_count}.mp4"
        
        return filename
    
    def _save_frames_as_video(self, frames, output_path):
        """フレームを動画ファイルとして保存"""
        try:
            import cv2
            
            if not frames:
                self.video_logger.warning("保存するフレームがありません")
                return
            
            # フレームの検証
            validated_frames = []
            for frame in frames:
                if frame is not None and len(frame.shape) == 3:
                    validated_frames.append(frame)
                else:
                    self.video_logger.debug("無効なフレームをスキップ")
            
            if not validated_frames:
                self.video_logger.warning("有効なフレームがありません")
                return
            
            # 動画パラメータ
            height, width, _ = validated_frames[0].shape
            fps = 50.0  # 50fps
            
            # VideoWriterの設定
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.video_logger.error(f"VideoWriterの初期化に失敗: {output_path}")
                return
            
            # フレームの書き込み
            written_frames = 0
            for i, frame in enumerate(validated_frames):
                try:
                    # BGR形式に変換（OpenCVはBGR形式）
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                    written_frames += 1
                except Exception as e:
                    self.video_logger.warning(f"フレーム {i} 書き込みエラー: {e}")
                    continue
            
            # リソースの解放
            out.release()
            
            self.video_logger.info(f"動画保存完了: {output_path}", {
                "total_frames": len(frames),
                "validated_frames": len(validated_frames),
                "written_frames": written_frames,
                "fps": fps,
                "resolution": f"{width}x{height}"
            })
            
        except ImportError:
            self.video_logger.warning("OpenCVがインストールされていません。動画保存をスキップします。")
        except Exception as e:
            self.video_logger.error("動画保存中にエラーが発生しました", exception=e)
            import traceback
            self.video_logger.debug(f"スタックトレース: {traceback.format_exc()}")
    
    def on_step(self):
        """各ステップでの処理（修正版）"""
        # 評価中のみフレームを取得
        if self.is_evaluating:
            try:
                # PyBulletからフレームを直接取得
                frame = self._get_pybullet_frame()
                if frame is not None:
                    self.frames.append(frame)
            except Exception as e:
                self.video_logger.debug("フレーム取得エラー", exception=e)
        
        return super().on_step()


class BittleTrainer:
    """Bittle学習管理クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        トレーナーの初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = get_logger("bittle_trainer")
        self.logger.info("=== Bittle学習システム初期化開始 ===")
        
        try:
            # 設定の読み込み
            self.config = self._load_config(config_path)
            
            # 既存のbest_modelをバックアップ（新しい学習セッション開始前）
            self._backup_previous_best_model()
            
            # ディレクトリの作成
            self._create_directories()
            
            # 環境の初期化
            self.env = None
            self.eval_env = None
            self.model = None
            
            # 学習統計
            self.training_stats = {
                'start_time': None,
                'end_time': None,
                'total_timesteps': 0,
                'episodes_completed': 0,
                'best_reward': float('-inf'),
                'training_history': []
            }
            
            self.logger.info("Bittle学習システム初期化完了")
            
        except Exception as e:
            self.logger.critical("学習システム初期化エラー", exception=e)
            raise
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定の読み込み"""
        if config_path is None:
            config_path = "configs/default.yaml"
        
        self.logger.info("設定ファイル読み込み", {"config_path": config_path})
        return load_and_validate_config(config_path)
    
    def _backup_previous_best_model(self):
        """
        既存のbest_modelをバックアップ
        
        新しい学習セッションを開始する前に、前回の学習で作成された
        best_model.zipをアーカイブディレクトリに保存します。
        これにより、報酬設定を変更して再学習する際も、過去の優秀な
        モデルを失わずに保存できます。
        """
        try:
            # best_modelのパスを構築
            best_model_dir = os.path.join(self.config['save']['model_path'], 'best_model')
            best_model_file = os.path.join(best_model_dir, 'best_model.zip')
            
            # ファイルが存在しない場合はバックアップ不要（初回学習）
            if not os.path.exists(best_model_file):
                self.logger.info("バックアップ対象のbest_modelが存在しません（初回学習）")
                return
            
            # バックアップディレクトリの作成
            backup_dir = os.path.join(self.config['save']['model_path'], 'best_model_archive')
            os.makedirs(backup_dir, exist_ok=True)
            
            # ファイルの最終更新日時を取得してタイムスタンプとする
            mtime = os.path.getmtime(best_model_file)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(mtime))
            
            # バックアップファイル名を生成
            backup_file = os.path.join(backup_dir, f'best_model_{timestamp}.zip')
            
            # 同名のバックアップが既に存在する場合の対策
            counter = 1
            original_backup_file = backup_file
            while os.path.exists(backup_file):
                # 連番を追加して一意なファイル名にする
                backup_file = original_backup_file.replace('.zip', f'_{counter}.zip')
                counter += 1
                if counter > 100:  # 無限ループ防止
                    self.logger.warning("バックアップファイル名の生成に失敗しました")
                    return
            
            # ファイルをコピー（メタデータも保持）
            shutil.copy2(best_model_file, backup_file)
            
            # ファイルサイズを確認（コピーが正常に完了したか検証）
            original_size = os.path.getsize(best_model_file)
            backup_size = os.path.getsize(backup_file)
            
            if original_size == backup_size:
                self.logger.info("前回のbest_modelをバックアップしました", {
                    "original_file": best_model_file,
                    "backup_file": backup_file,
                    "file_size": f"{backup_size / 1024:.1f} KB",
                    "timestamp": timestamp
                })
            else:
                self.logger.error("バックアップファイルのサイズが一致しません", {
                    "original_size": original_size,
                    "backup_size": backup_size
                })
                # サイズが一致しない場合は不完全なバックアップを削除
                os.remove(backup_file)
                
        except Exception as e:
            # バックアップに失敗しても学習は続行可能
            # ただし警告ログを出力してユーザーに通知
            self.logger.warning("best_modelのバックアップ中にエラーが発生しました", {
                "error": str(e),
                "note": "学習は続行されますが、前回のbest_modelが上書きされる可能性があります"
            })
    
    def _create_directories(self):
        """必要なディレクトリの作成"""
        directories = [
            self.config['logging']['log_dir'],
            self.config['save']['model_path'],
            self.config['save']['checkpoint_path'],
            os.path.dirname(self.config['save']['vec_normalize_path'])
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"ディレクトリ作成: {directory}")
    
    def create_environment(self) -> gym.Env:
        """学習用環境の作成"""
        self.logger.debug("学習環境作成開始")
        
        try:
            def make_env():
                return BittleEnvironment(self.config, render_mode=None)
            
            env = DummyVecEnv([make_env])
            
            # VecNormalizeの適用
            env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)
            
            self.logger.info("学習環境作成完了")
            return env
            
        except Exception as e:
            self.logger.error("環境作成エラー", exception=e)
            raise
    
    def create_evaluation_environment(self) -> gym.Env:
        """評価用環境の作成"""
        self.logger.debug("評価環境作成開始")
        
        try:
            def make_eval_env():
                # 動画生成のためrgb_arrayモードで作成
                render_mode = "rgb_array"
                if self.config['training'].get('n_envs', 1) == 1:
                    # 単一環境の場合はhumanモードも可能
                    render_mode = self.config['environment'].get('render_mode', "rgb_array")
                
                return BittleEnvironment(self.config, render_mode=render_mode)
            
            eval_env = DummyVecEnv([make_eval_env])
            # 学習環境と同じ正規化を適用
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False)
            
            self.logger.info("評価環境作成完了")
            return eval_env
            
        except Exception as e:
            self.logger.error("評価環境作成エラー", exception=e)
            raise
    
    def create_model(self, env: gym.Env) -> PPO:
        """モデルの作成"""
        self.logger.debug("PPOモデル作成開始")
        
        try:
            training_config = self.config['training']
            
            # TensorBoardログディレクトリの設定
            tb_log_dir = os.path.join(self.config['logging']['log_dir'], 'tensorboard')
            os.makedirs(tb_log_dir, exist_ok=True)
            
            # 既存の実行を確認してユニークなログ名を生成
            existing_runs = glob.glob(os.path.join(tb_log_dir, "bittle_training_*"))
            run_number = len(existing_runs) + 1
            tb_log_name = f"bittle_training_{run_number}"
            self.logger.info(f"TensorBoardログ名: {tb_log_name}")
            
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=training_config['learning_rate'],
                n_steps=training_config['n_steps'],
                batch_size=training_config['batch_size'],
                n_epochs=training_config['n_epochs'],
                gamma=training_config['gamma'],
                gae_lambda=training_config['gae_lambda'],
                clip_range=training_config['clip_range'],
                ent_coef=training_config['ent_coef'],
                vf_coef=training_config['vf_coef'],
                max_grad_norm=training_config['max_grad_norm'],
                tensorboard_log=tb_log_dir,
                verbose=1
            )
            
            self.logger.info("PPOモデル作成完了", {
                "learning_rate": training_config['learning_rate'],
                "batch_size": training_config['batch_size'],
                "tensorboard_log": tb_log_dir
            })
            
            return model
            
        except Exception as e:
            self.logger.error("モデル作成エラー", exception=e)
            raise
    
    def create_callbacks(self, eval_env: gym.Env) -> List[BaseCallback]:
        """コールバックの作成"""
        self.logger.debug("コールバック作成開始")
        
        callbacks = []
        
        try:
            # 動画保存機能付き評価コールバック
            eval_callback = VideoEvalCallback(
                eval_env,
                best_model_save_path=os.path.join(self.config['save']['model_path'], 'best_model'),
                log_path=self.config['logging']['log_dir'],
                eval_freq=self.config['evaluation']['frequency'],
                n_eval_episodes=self.config['evaluation']['n_eval_episodes'],
                deterministic=self.config['evaluation']['deterministic'],
                render=False,  # 評価時のレンダリング無効化（エラー回避）
                verbose=1
            )
            callbacks.append(eval_callback)
            
            # チェックポイントコールバック
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config['save']['frequency'],
                save_path=self.config['save']['checkpoint_path'],
                name_prefix='bittle_checkpoint'
            )
            callbacks.append(checkpoint_callback)
            
            # デバッグコールバック
            debug_callback = DebugCallback(verbose=1)
            callbacks.append(debug_callback)
            
            self.logger.info("コールバック作成完了", {
                "eval_callback": "VideoEvalCallback",
                "checkpoint_callback": "CheckpointCallback",
                "debug_callback": "DebugCallback"
            })
            
            return callbacks
            
        except Exception as e:
            self.logger.error("コールバック作成エラー", exception=e)
            raise
    
    def train(self) -> PPO:
        """学習の実行"""
        self.logger.info("=== 学習開始 ===")
        
        try:
            # 環境の作成
            self.env = self.create_environment()
            self.eval_env = self.create_evaluation_environment()
            
            # モデルの作成
            self.model = self.create_model(self.env)
            
            # コールバックの作成
            callbacks = self.create_callbacks(self.eval_env)
            
            # 学習統計の初期化
            self.training_stats['start_time'] = time.time()
            
            # 学習の実行
            total_timesteps = self.config['training']['total_timesteps']
            
            self.logger.info("学習実行開始", {
                "total_timesteps": total_timesteps,
                "eval_freq": self.config['evaluation']['frequency'],
                "save_freq": self.config['save']['frequency']
            })
            
            # TensorBoardログ名の設定
            existing_runs = glob.glob(os.path.join(self.config['logging']['log_dir'], 'tensorboard', "bittle_training_*"))
            run_number = len(existing_runs) + 1
            tb_log_name = f"bittle_training_{run_number}"
            self.logger.info(f"TensorBoardログ名: {tb_log_name}")
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=100,
                tb_log_name=tb_log_name,
                progress_bar=True
            )
            
            # 学習統計の更新
            self.training_stats['end_time'] = time.time()
            self.training_stats['total_timesteps'] = total_timesteps
            
            # 最終モデルの保存
            self._finalize_training()
            
            self.logger.info("学習完了", {
                "total_timesteps": total_timesteps,
                "training_time": self.training_stats['end_time'] - self.training_stats['start_time']
            })
            
            return self.model
            
        except Exception as e:
            self.logger.error("学習エラー", exception=e)
            raise
    
    def _finalize_training(self):
        """学習終了時の処理"""
        try:
            # 最終モデルの保存
            final_model_path = os.path.join(self.config['save']['model_path'], 'final_model.zip')
            self.model.save(final_model_path)
            
            # VecNormalizeの保存
            vec_normalize_path = self.config['save']['vec_normalize_path']
            self.env.save(vec_normalize_path)
            
            # 学習統計の保存
            stats_path = os.path.join(self.config['save']['model_path'], 'training_stats.json')
            import json
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
            
            self.logger.info("学習結果保存完了", {
                "final_model": final_model_path,
                "vec_normalize": vec_normalize_path,
                "training_stats": stats_path
            })
            
        except Exception as e:
            self.logger.error("学習結果保存エラー", exception=e)
            raise


def train_agent(config_path: str = "configs/default.yaml") -> PPO:
    """
    エージェントの学習
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        PPO: 学習済みモデル
    """
    trainer = BittleTrainer(config_path)
    return trainer.train()


def resume_training(checkpoint_path: str, config_path: str = "configs/default.yaml") -> PPO:
    """
    チェックポイントからの学習再開
    
    Args:
        checkpoint_path: チェックポイントファイルのパス
        config_path: 設定ファイルのパス
        
    Returns:
        PPO: 学習済みモデル
    """
    logger = get_logger("resume_training")
    logger.info("チェックポイントからの学習再開", {"checkpoint_path": checkpoint_path})
    
    try:
        if not os.path.exists(checkpoint_path):
            raise ModelLoadError(checkpoint_path, "Checkpoint file not found")
        
        # トレーナーの初期化
        trainer = BittleTrainer(config_path)
        
        # 環境の作成
        trainer.env = trainer.create_environment()
        trainer.eval_env = trainer.create_evaluation_environment()
        
        # チェックポイントからモデルを読み込み
        trainer.model = PPO.load(checkpoint_path, env=trainer.env)
        
        # 残りの学習を実行
        remaining_timesteps = trainer.config['training']['total_timesteps']
        callbacks = trainer.create_callbacks(trainer.eval_env)
        
        trainer.model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=False
        )
        
        trainer._finalize_training()
        return trainer.model
        
    except Exception as e:
        logger.error("学習再開エラー", exception=e)
        raise


if __name__ == "__main__":
    import argparse
    try:
        import torch
    except ImportError:
        torch = None
    
    parser = argparse.ArgumentParser(description='Bittle四足歩行ロボット学習')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='設定ファイルのパス')
    parser.add_argument('--resume', type=str, default=None,
                        help='再開するチェックポイントのパス')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグモードで実行')
    
    args = parser.parse_args()
    
    # デバッグレベルの設定
    if args.debug:
        os.environ['LOGGING_LEVEL'] = 'DEBUG'
    
    try:
        if args.resume:
            model = resume_training(args.resume, args.config)
        else:
            model = train_agent(args.config)
        
        print("学習が正常に完了しました！")
        
    except KeyboardInterrupt:
        print("学習が中断されました")
    except Exception as e:
        print(f"学習中にエラーが発生しました: {e}")
        raise