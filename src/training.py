"""
Bittle四足歩行ロボットの学習スクリプト
デバッグとモニタリングを重視した実装
"""

import yaml
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import json

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.logger import configure
import gymnasium as gym

from .environment import BittleEnvironment
from .utils.logger import get_logger
from .utils.config_validator import load_and_validate_config, create_default_config
from .utils.exceptions import ModelLoadError, ConfigValidationError


class DebugCallback(BaseCallback):
    """デバッグ用カスタムコールバック"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.debug_logger = get_logger("training_callback")
        self.episode_rewards = []
        self.episode_lengths = []
        self.performance_history = []
        
    def _on_step(self) -> bool:
        """各ステップで呼ばれるコールバック"""
        # 基本統計の収集
        if self.locals.get('done'):
            episode_reward = self.locals.get('episode_reward', 0)
            episode_length = self.locals.get('episode_length', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # シンプルなログ出力（統計計算はTensorBoardに任せる）
            self.debug_logger.info("エピソード完了", {
                "episode": len(self.episode_rewards),
                "episode_reward": episode_reward,
                "episode_length": episode_length
            })
        
        return True
    
    def _on_training_end(self) -> None:
        """学習終了時のコールバック"""
        self.debug_logger.info("学習完了統計", {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": np.max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": np.min(self.episode_rewards) if self.episode_rewards else 0
        })


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
        """設定の読み込み（安全版）"""
        if config_path is None:
            self.logger.warning("設定ファイルが指定されていません。")
            self.logger.info("デフォルト設定を使用します。")
            self.logger.info("推奨: python -m src.training --config configs/default.yaml")
            return create_default_config()
        
        if not os.path.exists(config_path):
            self.logger.error(f"設定ファイルが見つかりません: {config_path}")
            self.logger.critical("学習を停止します。正しい設定ファイルを指定してください。")
            self.logger.info("利用可能な設定ファイル:")
            self.logger.info("  - configs/default.yaml")
            self.logger.info("  - configs/experiment.yaml") 
            self.logger.info("  - configs/production.yaml")
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        try:
            config = load_and_validate_config(config_path)
            self.logger.info("設定ファイルの読み込み完了", {"config_path": config_path})
            
            # 設定の要約をログ出力
            self._log_config_summary(config)
            return config
            
        except Exception as e:
            self.logger.error("設定ファイル読み込みエラー", exception=e)
            self.logger.critical("学習を停止します。設定ファイルを修正してください。")
            self.logger.info("設定ファイルの構文を確認してください。")
            raise ConfigValidationError(f"設定ファイルの読み込みに失敗しました: {config_path}") from e
    
    def _log_config_summary(self, config: Dict[str, Any]):
        """設定の要約をログ出力"""
        self.logger.info("=== 学習設定の要約 ===", {
            "total_timesteps": config['training']['total_timesteps'],
            "algorithm": config['training']['algorithm'],
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['batch_size'],
            "max_episode_steps": config['environment']['max_episode_steps'],
            "n_envs": config['training'].get('n_envs', 1)
        })
        
        # 本格学習用設定の警告
        if config['training']['total_timesteps'] > 500000:
            self.logger.warning("本格的な学習設定が検出されました。")
            self.logger.warning("学習時間が長くなる可能性があります。")
    
    def _create_directories(self):
        """必要なディレクトリの作成"""
        dirs_to_create = [
            self.config['logging']['log_dir'],
            self.config['save']['model_path'],
            self.config['save']['checkpoint_path'],
            'models/vec_normalize',
            'models/eval'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"ディレクトリ作成: {dir_path}")
    
    def create_environment(self) -> gym.Env:
        """環境の作成"""
        self.logger.debug("環境作成開始")
        
        try:
            # シングル環境の作成（デバッグしやすくするため）
            def make_env():
                return BittleEnvironment(self.config)
            
            # マルチプロセス環境の作成
            n_envs = self.config['training'].get('n_envs', 12)
            if n_envs > 1:
                self.logger.info(f"マルチプロセス環境を作成 (n_envs={n_envs})")
                env = make_vec_env(make_env, n_envs=n_envs)
            else:
                self.logger.info("シングルプロセス環境を作成")
                env = DummyVecEnv([make_env])
            
            # 環境の正規化
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0
            )
            
            self.logger.info("環境作成完了")
            return env
            
        except Exception as e:
            self.logger.error("環境作成エラー", exception=e)
            raise
    
    def create_evaluation_environment(self) -> gym.Env:
        """評価用環境の作成"""
        self.logger.debug("評価環境作成開始")
        
        try:
            def make_eval_env():
                return BittleEnvironment(self.config, render_mode=None)
            
            eval_env = DummyVecEnv([make_eval_env])
            # 学習環境と同じ正規化を適用
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
            
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
            
            # TensorBoardロガーの設定
            tb_log_dir = os.path.join(self.config['logging']['log_dir'], 'tensorboard')
            Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
            
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=training_config['learning_rate'],
                n_steps=training_config.get('n_steps', 2048),
                batch_size=training_config['batch_size'],
                n_epochs=training_config['n_epochs'],
                gamma=training_config['gamma'],
                gae_lambda=training_config['gae_lambda'],
                clip_range=training_config['clip_range'],
                ent_coef=training_config['ent_coef'],
                vf_coef=training_config['vf_coef'],
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
                ),
                verbose=self.config['logging'].get('verbose', 1),
                tensorboard_log=tb_log_dir,
                device='auto'
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
            # 評価コールバック
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(self.config['save']['model_path'], 'best_model'),
                log_path=self.config['logging']['log_dir'],
                eval_freq=self.config['evaluation']['frequency'],
                n_eval_episodes=self.config['evaluation']['n_eval_episodes'],
                deterministic=self.config['evaluation']['deterministic'],
                render=False,
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
                "eval_frequency": self.config['evaluation']['frequency'],
                "save_frequency": self.config['save']['frequency']
            })
            
            return callbacks
            
        except Exception as e:
            self.logger.error("コールバック作成エラー", exception=e)
            raise
    
    def train(self) -> PPO:
        """学習の実行"""
        self.logger.info("=== 学習開始 ===")
        
        try:
            # 学習開始時間の記録
            self.training_stats['start_time'] = time.time()
            
            # 環境の作成
            self.logger.info("環境の初期化...")
            self.env = self.create_environment()
            self.eval_env = self.create_evaluation_environment()
            
            # モデルの作成
            self.logger.info("モデルの初期化...")
            self.model = self.create_model(self.env)
            
            # コールバックの作成
            self.logger.info("コールバックの初期化...")
            callbacks = self.create_callbacks(self.eval_env)
            
            # 学習パラメータのログ
            total_timesteps = self.config['training']['total_timesteps']
            self.logger.info("学習パラメータ", {
                "total_timesteps": total_timesteps,
                "algorithm": self.config['training']['algorithm'],
                "learning_rate": self.config['training']['learning_rate'],
                "batch_size": self.config['training']['batch_size']
            })
            
            # 学習の実行
            self.logger.info("学習実行開始...")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=100,
                tb_log_name="bittle_training",
                progress_bar=True
            )
            
            # 学習完了処理
            self._finalize_training()
            
            return self.model
            
        except Exception as e:
            self.logger.error("学習中にエラーが発生しました", exception=e)
            raise
        finally:
            self._cleanup()
    
    def _finalize_training(self):
        """学習完了処理"""
        self.training_stats['end_time'] = time.time()
        training_duration = self.training_stats['end_time'] - self.training_stats['start_time']
        
        # 最終モデルの保存
        final_model_path = os.path.join(self.config['save']['model_path'], 'final_model')
        self.model.save(final_model_path)
        
        # VecNormalizeの保存
        vec_normalize_path = self.config['save']['vec_normalize_path']
        self.env.save(vec_normalize_path)
        
        # 学習統計の保存
        self._save_training_stats(training_duration)
        
        self.logger.info("=== 学習完了 ===", {
            "training_duration": f"{training_duration:.2f}秒",
            "final_model_path": final_model_path,
            "vec_normalize_path": vec_normalize_path
        })
    
    def _save_training_stats(self, training_duration: float):
        """学習統計の保存"""
        stats = {
            'training_config': self.config['training'],
            'training_duration_seconds': training_duration,
            'total_timesteps': self.config['training']['total_timesteps'],
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment_config': self.config['environment'],
            'robot_config': self.config['robot'],
            'reward_config': self.config['rewards']
        }
        
        stats_path = os.path.join(self.config['save']['model_path'], 'training_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info("学習統計を保存しました", {"stats_path": stats_path})
    
    def _cleanup(self):
        """リソースのクリーンアップ"""
        self.logger.debug("リソースクリーンアップ開始")
        
        try:
            if self.env is not None:
                self.env.close()
            if self.eval_env is not None:
                self.eval_env.close()
            
            self.logger.debug("リソースクリーンアップ完了")
            
        except Exception as e:
            self.logger.error("クリーンアップエラー", exception=e)


def train_agent(config_path: str = "configs/default.yaml") -> PPO:
    """
    エージェントの学習実行（便利関数）
    
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
