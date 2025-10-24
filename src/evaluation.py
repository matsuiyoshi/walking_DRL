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
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from .environment import BittleEnvironment
from .utils.logger import get_logger
from .utils.config_validator import load_and_validate_config
from .utils.exceptions import ModelLoadError


class BittleEvaluator:
    """Bittle評価管理クラス"""
    
    def __init__(self, model_path: str, config_path: str = "configs/production.yaml"):
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
            # 環境の作成（動画生成のためrgb_arrayモードで作成）
            def make_env():
                return BittleEnvironment(self.config, render_mode="rgb_array")
            
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
    
    def evaluate_model(self, num_episodes: int = 3, deterministic: bool = True, 
                      render: bool = False, save_video: bool = True) -> Dict:
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
        
        # 動画保存用のフレームリスト
        frames = []
        
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
            
            # 動画保存用フレーム取得（修正版）
            if save_video:
                try:
                    # PyBulletから直接フレームを取得
                    frame = self._get_pybullet_frame()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    self.logger.debug(f"フレーム取得エラー: {e}")
            
            # レンダリング（修正版）
            if render:
                # 実際の可視化処理は環境のrender_mode="human"で行う
                time.sleep(0.02)  # 50Hz相当
            
            # 終了判定
            if done[0] if isinstance(done, np.ndarray) else done:
                break
        
        episode_duration = time.time() - episode_start_time
        
        # 動画保存
        video_path = None
        if save_video and frames:
            video_path = self._save_episode_video(frames, episode_num)
        
        episode_result = {
            'episode_num': episode_num,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_duration': episode_duration,
            'mean_forward_velocity': float(np.mean(forward_velocities)) if forward_velocities else 0.0,
            'mean_robot_height': float(np.mean(robot_heights)) if robot_heights else 0.0,
            'energy_consumption': energy_consumption,
            'forward_velocities': [float(v) for v in forward_velocities],
            'robot_heights': [float(h) for h in robot_heights],
            'actions_taken': [[float(a) for a in action] for action in actions_taken],
            'video_path': video_path
        }
        
        self.logger.debug(f"エピソード {episode_num + 1} 完了", {
            "reward": float(episode_reward),
            "length": int(episode_length),
            "duration": f"{episode_duration:.2f}s",
            "video_path": video_path
        })
        
        return episode_result
    
    def _get_pybullet_frame(self):
        """PyBullet環境からフレームを取得（修正版）"""
        try:
            import pybullet as p
            
            # 環境からロボット情報を取得
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                base_env = self.env.envs[0]
                
                # VecNormalizeでラップされている場合の正しい構造
                if hasattr(base_env, 'env'):
                    actual_env = base_env.env
                else:
                    actual_env = base_env
                
                # BittleEnvironmentからロボット情報を取得
                if hasattr(actual_env, 'robot_id') and hasattr(actual_env, 'physics_client'):
                    robot_id = actual_env.robot_id
                    physics_client = actual_env.physics_client
                    
                    if robot_id is None or physics_client is None:
                        self.logger.debug("ロボットIDまたは物理クライアントがNone")
                        return None
                    
                    # カメラパラメータの設定
                    width, height = 640, 480
                    
                    # ロボットの現在位置を取得
                    position, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
                    
                    # ビューマトリックスの計算
                    view_matrix = p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=position,
                        distance=1.5,
                        yaw=45,
                        pitch=-30,
                        roll=0,
                        upAxisIndex=2,
                        physicsClientId=physics_client
                    )
                    
                    # プロジェクションマトリックスの計算
                    projection_matrix = p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=width/height,
                        nearVal=0.1,
                        farVal=100,
                        physicsClientId=physics_client
                    )
                    
                    # カメラ画像の取得
                    _, _, rgb_array, _, _ = p.getCameraImage(
                        width=width,
                        height=height,
                        viewMatrix=view_matrix,
                        projectionMatrix=projection_matrix,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                        physicsClientId=physics_client
                    )
                    
                    # RGB配列を適切な形状に変換
                    rgb_array = np.array(rgb_array).reshape(height, width, 4)[:, :, :3]
                    
                    self.logger.debug(f"フレーム取得成功: {rgb_array.shape}")
                    return rgb_array.astype(np.uint8)
            
            self.logger.debug("環境からロボット情報を取得できませんでした")
            return None
            
        except Exception as e:
            self.logger.debug(f"PyBulletフレーム取得エラー: {e}")
            return None
    
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
            # float32をfloatに変換してJSONシリアライゼーション可能にする
            def convert_float32(obj):
                if isinstance(obj, dict):
                    return {k: convert_float32(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_float32(item) for item in obj]
                elif hasattr(obj, 'dtype') and 'float32' in str(obj.dtype):
                    return float(obj)
                else:
                    return obj
            
            converted_result = convert_float32(analysis_result)
            json.dump(converted_result, f, indent=2, ensure_ascii=False)
        
        # グラフの作成と保存
        self._create_evaluation_plots(analysis_result, results_dir, timestamp)
        
        self.logger.info("評価結果保存完了", {
            "json_path": str(json_path),
            "results_dir": str(results_dir)
        })
    
    def _save_episode_video(self, frames: List[np.ndarray], episode_num: int) -> str:
        """エピソード動画の保存"""
        try:
            # recordingsディレクトリの作成（絶対パス）
            recordings_dir = '/app/recordings'
            os.makedirs(recordings_dir, exist_ok=True)
            
            # 動画ファイル名の生成
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_filename = f'{recordings_dir}/bittle_walking_episode_{episode_num + 1}_{timestamp}.mp4'
            
            if not frames:
                self.logger.warning(f"エピソード {episode_num + 1}: 保存するフレームがありません")
                return None
            
            # フレームサイズの取得
            height, width, _ = frames[0].shape
            
            # 動画書き込み設定
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 50.0  # 50fps
            
            # VideoWriterの初期化
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            
            # フレームを動画に書き込み
            for frame in frames:
                # RGBからBGRに変換（OpenCVはBGR形式）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # VideoWriterを解放
            out.release()
            
            self.logger.info(f"動画保存完了: {video_filename}", {
                "episode": episode_num + 1,
                "frame_count": len(frames),
                "video_size": f"{width}x{height}",
                "fps": fps
            })
            
            return video_filename
            
        except Exception as e:
            self.logger.error(f"動画保存エラー (エピソード {episode_num + 1}): {e}")
            return None
    
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
            # 可視化環境の作成（修正版）
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
                    obs, reward, terminated, truncated, info = visual_env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward[0]
                    episode_steps += 1
                    
                    # レンダリング（修正版）
                    # render_mode="human"で環境を作成しているため、自動的にGUI表示される
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


def evaluate_model(model_path: str, config_path: str = "configs/production.yaml", 
                  num_episodes: int = 3, render: bool = False, save_video: bool = True) -> Dict:
    """
    モデル評価の便利関数
    
    Args:
        model_path: 学習済みモデルのパス
        config_path: 設定ファイルのパス
        num_episodes: 評価エピソード数
        render: 可視化するかどうか
        save_video: 動画保存するかどうか
        
    Returns:
        Dict: 評価結果
    """
    evaluator = BittleEvaluator(model_path, config_path)
    return evaluator.evaluate_model(num_episodes, render=render, save_video=save_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bittle四足歩行ロボット評価')
    parser.add_argument('model_path', type=str, help='学習済みモデルのパス')
    parser.add_argument('--config', type=str, default='configs/production.yaml',
                        help='設定ファイルのパス')
    parser.add_argument('--episodes', type=int, default=3,
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
                render=args.render,
                save_video=True  # 動画保存を有効化
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
