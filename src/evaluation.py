"""
Bittle四足歩行ロボットの評価スクリプト
学習済みモデルの性能評価とデバッグ支援
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from .environment import BittleEnvironment
from .utils.logger import get_logger
from .utils.config_validator import load_and_validate_config
from .utils.exceptions import ModelLoadError


class BittleEvaluator:
    """Bittle評価管理クラス"""
    
    def __init__(self, model_path: str, config_path: str = "configs/default.yaml"):
        """
        評価器の初期化
        
        Args:
            model_path: 学習済みモデルのパス
            config_path: 設定ファイルのパス
        """
        self.logger = get_logger("bittle_evaluator")
        self.logger.info("=== Bittle評価システム初期化開始 ===")
        
        try:
            # 設定の読み込み
            self.config = load_and_validate_config(config_path)
            
            # モデルとファイルパスの設定
            self.model_path = model_path
            self.vec_normalize_path = self.config['save']['vec_normalize_path']
            
            # 環境とモデルの初期化
            self.env = None
            self.model = None
            
            # 評価結果の保存
            self.evaluation_results = {
                'episode_rewards': [],
                'episode_lengths': [],
                'forward_velocities': [],
                'robot_heights': [],
                'energy_consumptions': [],
                'detailed_logs': []
            }
            
            self.logger.info("Bittle評価システム初期化完了")
            
        except Exception as e:
            self.logger.critical("評価システム初期化エラー", exception=e)
            raise
    
    def load_model_and_environment(self):
        """モデルと環境の読み込み"""
        self.logger.info("モデルと環境の読み込み開始")
        
        try:
            # 環境の作成
            def make_env():
                return BittleEnvironment(self.config, render_mode=None)
            
            self.env = DummyVecEnv([make_env])
            
            # VecNormalizeの読み込み（存在する場合）
            if os.path.exists(self.vec_normalize_path):
                self.env = VecNormalize.load(self.vec_normalize_path, self.env)
                self.env.training = False  # 評価モード
                self.env.norm_reward = False  # 評価時は報酬正規化無効
                self.logger.info("VecNormalize読み込み完了")
            else:
                self.logger.warning(f"VecNormalizeファイルが見つかりません: {self.vec_normalize_path}")
            
            # モデルの読み込み
            if not os.path.exists(self.model_path):
                raise ModelLoadError(self.model_path)
            
            self.model = PPO.load(self.model_path, env=self.env)
            
            self.logger.info("モデルと環境の読み込み完了", {
                "model_path": self.model_path,
                "vec_normalize_path": self.vec_normalize_path
            })
            
        except Exception as e:
            self.logger.error("モデル・環境読み込みエラー", exception=e)
            raise
    
    def evaluate_model(self, num_episodes: int = 10, deterministic: bool = True, 
                      render: bool = False, save_video: bool = False) -> Dict:
        """
        モデルの評価実行
        
        Args:
            num_episodes: 評価エピソード数
            deterministic: 決定的行動かどうか
            render: 可視化するかどうか
            save_video: 動画保存するかどうか
            
        Returns:
            Dict: 評価結果
        """
        self.logger.info("モデル評価開始", {
            "num_episodes": num_episodes,
            "deterministic": deterministic,
            "render": render
        })
        
        try:
            # モデルと環境の読み込み
            if self.model is None or self.env is None:
                self.load_model_and_environment()
            
            # 評価ループ
            for episode in range(num_episodes):
                episode_result = self._evaluate_single_episode(
                    episode, deterministic, render, save_video
                )
                self._store_episode_result(episode_result)
            
            # 結果の分析
            analysis_result = self._analyze_results()
            
            # 結果の保存
            self._save_evaluation_results(analysis_result)
            
            self.logger.info("モデル評価完了", analysis_result['summary'])
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("モデル評価エラー", exception=e)
            raise
        finally:
            if self.env:
                self.env.close()
    
    def _evaluate_single_episode(self, episode_num: int, deterministic: bool, 
                                render: bool, save_video: bool) -> Dict:
        """単一エピソードの評価"""
        episode_start_time = time.time()
        self.logger.debug(f"エピソード {episode_num + 1} 開始")
        
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        forward_velocities = []
        robot_heights = []
        actions_taken = []
        energy_consumption = 0
        
        while True:
            # アクションの予測
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # アクションの実行
            obs, reward, done, info = self.env.step(action)
            
            # 統計の収集
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            actions_taken.append(action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action)
            
            # 詳細情報の収集（環境から取得可能な場合）
            if isinstance(info, list) and len(info) > 0:
                episode_info = info[0]
                if 'forward_velocity' in episode_info:
                    forward_velocities.append(episode_info['forward_velocity'])
                if 'robot_height' in episode_info:
                    robot_heights.append(episode_info['robot_height'])
            
            # エネルギー消費の計算
            if isinstance(action, np.ndarray):
                energy_consumption += np.sum(np.abs(action))
            
            # レンダリング
            if render:
                self.env.render()
                time.sleep(0.02)  # 50Hz相当
            
            # 終了判定
            if done[0] if isinstance(done, np.ndarray) else done:
                break
        
        episode_duration = time.time() - episode_start_time
        
        episode_result = {
            'episode_num': episode_num,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_duration': episode_duration,
            'mean_forward_velocity': np.mean(forward_velocities) if forward_velocities else 0,
            'mean_robot_height': np.mean(robot_heights) if robot_heights else 0,
            'energy_consumption': energy_consumption,
            'forward_velocities': forward_velocities,
            'robot_heights': robot_heights,
            'actions_taken': actions_taken
        }
        
        self.logger.debug(f"エピソード {episode_num + 1} 完了", {
            "reward": episode_reward,
            "length": episode_length,
            "duration": f"{episode_duration:.2f}s"
        })
        
        return episode_result
    
    def _store_episode_result(self, episode_result: Dict):
        """エピソード結果の保存"""
        self.evaluation_results['episode_rewards'].append(episode_result['episode_reward'])
        self.evaluation_results['episode_lengths'].append(episode_result['episode_length'])
        self.evaluation_results['forward_velocities'].append(episode_result['mean_forward_velocity'])
        self.evaluation_results['robot_heights'].append(episode_result['mean_robot_height'])
        self.evaluation_results['energy_consumptions'].append(episode_result['energy_consumption'])
        self.evaluation_results['detailed_logs'].append(episode_result)
    
    def _analyze_results(self) -> Dict:
        """評価結果の分析"""
        self.logger.debug("評価結果の分析開始")
        
        rewards = np.array(self.evaluation_results['episode_rewards'])
        lengths = np.array(self.evaluation_results['episode_lengths'])
        velocities = np.array(self.evaluation_results['forward_velocities'])
        heights = np.array(self.evaluation_results['robot_heights'])
        energies = np.array(self.evaluation_results['energy_consumptions'])
        
        analysis = {
            'summary': {
                'total_episodes': len(rewards),
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'mean_episode_length': float(np.mean(lengths)),
                'std_episode_length': float(np.std(lengths)),
                'mean_forward_velocity': float(np.mean(velocities)),
                'std_forward_velocity': float(np.std(velocities)),
                'mean_robot_height': float(np.mean(heights)),
                'mean_energy_consumption': float(np.mean(energies))
            },
            'detailed_results': self.evaluation_results,
            'success_metrics': self._calculate_success_metrics(rewards, velocities, lengths)
        }
        
        return analysis
    
    def _calculate_success_metrics(self, rewards: np.ndarray, velocities: np.ndarray, 
                                  lengths: np.ndarray) -> Dict:
        """成功指標の計算"""
        # 成功基準の定義
        min_reward_threshold = 50.0  # 最低報酬
        min_velocity_threshold = 0.05  # 最低前進速度 (m/s)
        min_survival_steps = 250  # 最低生存ステップ数
        
        successful_episodes = 0
        high_velocity_episodes = 0
        long_survival_episodes = 0
        
        for i in range(len(rewards)):
            if rewards[i] >= min_reward_threshold:
                successful_episodes += 1
            if velocities[i] >= min_velocity_threshold:
                high_velocity_episodes += 1
            if lengths[i] >= min_survival_steps:
                long_survival_episodes += 1
        
        return {
            'success_rate': successful_episodes / len(rewards),
            'high_velocity_rate': high_velocity_episodes / len(rewards),
            'long_survival_rate': long_survival_episodes / len(rewards),
            'successful_episodes': successful_episodes,
            'high_velocity_episodes': high_velocity_episodes,
            'long_survival_episodes': long_survival_episodes,
            'thresholds': {
                'min_reward': min_reward_threshold,
                'min_velocity': min_velocity_threshold,
                'min_survival_steps': min_survival_steps
            }
        }
    
    def _save_evaluation_results(self, analysis_result: Dict):
        """評価結果の保存"""
        # 結果ディレクトリの作成
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で詳細結果を保存
        json_path = results_dir / f"evaluation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        # グラフの作成と保存
        self._create_evaluation_plots(analysis_result, results_dir, timestamp)
        
        self.logger.info("評価結果保存完了", {
            "json_path": str(json_path),
            "results_dir": str(results_dir)
        })
    
    def _create_evaluation_plots(self, analysis_result: Dict, results_dir: Path, timestamp: str):
        """評価結果のグラフ作成"""
        try:
            detailed_results = analysis_result['detailed_results']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Bittle Walking Evaluation Results', fontsize=16)
            
            # エピソード報酬の推移
            axes[0, 0].plot(detailed_results['episode_rewards'], 'b-', marker='o')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # 前進速度の分布
            axes[0, 1].hist(detailed_results['forward_velocities'], bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('Forward Velocity Distribution')
            axes[0, 1].set_xlabel('Forward Velocity (m/s)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
            
            # エピソード長の推移
            axes[1, 0].plot(detailed_results['episode_lengths'], 'r-', marker='s')
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].grid(True)
            
            # エネルギー消費の推移
            axes[1, 1].plot(detailed_results['energy_consumptions'], 'purple', marker='^')
            axes[1, 1].set_title('Energy Consumption')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Energy')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 画像の保存
            plot_path = results_dir / f"evaluation_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.debug("評価グラフ作成完了", {"plot_path": str(plot_path)})
            
        except Exception as e:
            self.logger.error("評価グラフ作成エラー", exception=e)
    
    def interactive_evaluation(self):
        """インタラクティブ評価（GUI表示）"""
        self.logger.info("インタラクティブ評価開始")
        
        try:
            # 可視化環境の作成
            def make_visual_env():
                return BittleEnvironment(self.config, render_mode="human")
            
            visual_env = DummyVecEnv([make_visual_env])
            
            if os.path.exists(self.vec_normalize_path):
                visual_env = VecNormalize.load(self.vec_normalize_path, visual_env)
                visual_env.training = False
                visual_env.norm_reward = False
            
            # モデルの読み込み
            model = PPO.load(self.model_path, env=visual_env)
            
            self.logger.info("可視化環境準備完了。ESCキーで終了してください。")
            
            while True:
                obs = visual_env.reset()
                episode_reward = 0
                episode_steps = 0
                
                print(f"\n=== 新しいエピソード開始 ===")
                
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = visual_env.step(action)
                    
                    episode_reward += reward[0]
                    episode_steps += 1
                    
                    visual_env.render()
                    time.sleep(0.02)
                    
                    if done[0]:
                        print(f"エピソード終了: 報酬={episode_reward:.2f}, ステップ数={episode_steps}")
                        break
                
                # 次のエピソードを続けるかの確認
                user_input = input("次のエピソードを実行しますか？ (y/n): ")
                if user_input.lower() != 'y':
                    break
            
            visual_env.close()
            
        except Exception as e:
            self.logger.error("インタラクティブ評価エラー", exception=e)
            raise


def evaluate_model(model_path: str, config_path: str = "configs/default.yaml", 
                  num_episodes: int = 10, render: bool = False) -> Dict:
    """
    モデル評価の便利関数
    
    Args:
        model_path: 学習済みモデルのパス
        config_path: 設定ファイルのパス
        num_episodes: 評価エピソード数
        render: 可視化するかどうか
        
    Returns:
        Dict: 評価結果
    """
    evaluator = BittleEvaluator(model_path, config_path)
    return evaluator.evaluate_model(num_episodes, render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bittle四足歩行ロボット評価')
    parser.add_argument('model_path', type=str, help='学習済みモデルのパス')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='設定ファイルのパス')
    parser.add_argument('--episodes', type=int, default=10,
                        help='評価エピソード数')
    parser.add_argument('--render', action='store_true',
                        help='可視化して実行')
    parser.add_argument('--interactive', action='store_true',
                        help='インタラクティブ評価モード')
    
    args = parser.parse_args()
    
    try:
        evaluator = BittleEvaluator(args.model_path, args.config)
        
        if args.interactive:
            evaluator.interactive_evaluation()
        else:
            results = evaluator.evaluate_model(
                num_episodes=args.episodes,
                render=args.render
            )
            
            print("\n=== 評価結果サマリー ===")
            summary = results['summary']
            print(f"平均報酬: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
            print(f"平均エピソード長: {summary['mean_episode_length']:.1f} ± {summary['std_episode_length']:.1f}")
            print(f"平均前進速度: {summary['mean_forward_velocity']:.3f} ± {summary['std_forward_velocity']:.3f} m/s")
            print(f"成功率: {results['success_metrics']['success_rate']:.1%}")
            
    except KeyboardInterrupt:
        print("評価が中断されました")
    except Exception as e:
        print(f"評価中にエラーが発生しました: {e}")
        raise
