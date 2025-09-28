"""
BittleEnvironmentのテストスイート
デバッグとバリデーションを支援
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import BittleEnvironment
from src.utils.exceptions import EnvironmentError, URDFLoadError, ConfigValidationError
from src.utils.config_validator import create_default_config


class TestBittleEnvironment(unittest.TestCase):
    """BittleEnvironmentのテストクラス"""
    
    def setUp(self):
        """テストの初期化"""
        self.config = create_default_config()
        self.temp_dir = tempfile.mkdtemp()
        
        # テスト用の簡単な設定
        self.config['logging']['log_dir'] = os.path.join(self.temp_dir, 'logs')
        self.config['environment']['max_episode_steps'] = 100  # テストを高速化
        
        # PyBulletを使わないモックテスト用フラグ
        self.use_mock = os.environ.get('BITTLE_USE_MOCK', 'false').lower() == 'true'
        
    def tearDown(self):
        """テストの後処理"""
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except:
                pass
        
        # 一時ディレクトリの削除
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_validation(self):
        """設定検証のテスト"""
        print("設定検証テスト実行中...")
        
        # 正常な設定のテスト
        try:
            from src.utils.config_validator import validate_config
            validate_config(self.config)
            print("✓ 正常な設定の検証に成功")
        except Exception as e:
            self.fail(f"正常な設定の検証に失敗: {e}")
        
        # 不正な設定のテスト
        invalid_config = self.config.copy()
        invalid_config['environment']['control_frequency'] = 1000
        invalid_config['environment']['physics_frequency'] = 100  # 制御周波数より小さい
        
        with self.assertRaises(ConfigValidationError):
            validate_config(invalid_config)
        print("✓ 不正な設定の検証に成功")
    
    def test_environment_creation(self):
        """環境作成のテスト"""
        if self.use_mock:
            self.skipTest("モックモードでは環境作成テストをスキップ")
        
        print("環境作成テスト実行中...")
        
        try:
            self.env = BittleEnvironment(self.config)
            self.assertIsNotNone(self.env)
            print("✓ 環境作成に成功")
        except FileNotFoundError as e:
            self.skipTest(f"URDFファイルが見つからないため、テストをスキップ: {e}")
        except Exception as e:
            self.fail(f"環境作成に失敗: {e}")
    
    def test_action_space_validation(self):
        """行動空間の検証テスト"""
        if self.use_mock:
            self.skipTest("モックモードでは行動空間テストをスキップ")
        
        print("行動空間検証テスト実行中...")
        
        try:
            self.env = BittleEnvironment(self.config)
            
            # 行動空間の形状確認
            self.assertEqual(self.env.action_space.shape, (8,))
            print("✓ 行動空間の形状確認成功")
            
            # 有効なアクションのテスト
            valid_action = np.zeros(8)
            self.assertTrue(self.env.action_space.contains(valid_action))
            print("✓ 有効なアクションの確認成功")
            
            # 無効なアクションのテスト
            invalid_action = np.ones(9)  # 間違った次元
            self.assertFalse(self.env.action_space.contains(invalid_action))
            print("✓ 無効なアクションの確認成功")
            
        except FileNotFoundError:
            self.skipTest("URDFファイルが見つからないため、行動空間テストをスキップ")
        except Exception as e:
            self.fail(f"行動空間検証に失敗: {e}")
    
    def test_observation_space_validation(self):
        """観測空間の検証テスト"""
        if self.use_mock:
            self.skipTest("モックモードでは観測空間テストをスキップ")
        
        print("観測空間検証テスト実行中...")
        
        try:
            self.env = BittleEnvironment(self.config)
            
            # 観測空間の形状確認
            self.assertEqual(self.env.observation_space.shape, (22,))
            print("✓ 観測空間の形状確認成功")
            
            # 観測空間のサンプリングテスト
            sample_obs = self.env.observation_space.sample()
            self.assertTrue(self.env.observation_space.contains(sample_obs))
            print("✓ 観測空間のサンプリング成功")
            
        except FileNotFoundError:
            self.skipTest("URDFファイルが見つからないため、観測空間テストをスキップ")
        except Exception as e:
            self.fail(f"観測空間検証に失敗: {e}")
    
    def test_reset_functionality(self):
        """リセット機能のテスト"""
        if self.use_mock:
            self.skipTest("モックモードではリセットテストをスキップ")
        
        print("リセット機能テスト実行中...")
        
        try:
            self.env = BittleEnvironment(self.config)
            
            # リセット実行
            obs, info = self.env.reset()
            
            # 観測の確認
            self.assertEqual(len(obs), 22)
            self.assertIsInstance(obs, np.ndarray)
            print("✓ リセット後の観測確認成功")
            
            # メタ情報の確認
            self.assertIsInstance(info, dict)
            self.assertIn('episode_steps', info)
            self.assertEqual(info['episode_steps'], 0)
            print("✓ リセット後のメタ情報確認成功")
            
        except FileNotFoundError:
            self.skipTest("URDFファイルが見つからないため、リセットテストをスキップ")
        except Exception as e:
            self.fail(f"リセット機能テストに失敗: {e}")
    
    def test_step_functionality(self):
        """ステップ機能のテスト"""
        if self.use_mock:
            self.skipTest("モックモードではステップテストをスキップ")
        
        print("ステップ機能テスト実行中...")
        
        try:
            self.env = BittleEnvironment(self.config)
            obs, _ = self.env.reset()
            
            # アクションの実行
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 戻り値の確認
            self.assertEqual(len(obs), 22)
            self.assertIsInstance(reward, (float, np.floating))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            print("✓ ステップ実行の戻り値確認成功")
            
            # メタ情報の確認
            self.assertIn('episode_steps', info)
            self.assertEqual(info['episode_steps'], 1)
            print("✓ ステップ後のメタ情報確認成功")
            
        except FileNotFoundError:
            self.skipTest("URDFファイルが見つからないため、ステップテストをスキップ")
        except Exception as e:
            self.fail(f"ステップ機能テストに失敗: {e}")
    
    def test_episode_completion(self):
        """エピソード完了のテスト"""
        if self.use_mock:
            self.skipTest("モックモードではエピソード完了テストをスキップ")
        
        print("エピソード完了テスト実行中...")
        
        try:
            self.env = BittleEnvironment(self.config)
            obs, _ = self.env.reset()
            
            max_steps = self.config['environment']['max_episode_steps']
            step_count = 0
            
            while step_count < max_steps + 10:  # 安全マージン
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                step_count += 1
                
                if terminated or truncated:
                    break
            
            # エピソード終了の確認
            self.assertTrue(terminated or truncated)
            print(f"✓ エピソード完了確認成功 (ステップ数: {step_count})")
            
        except FileNotFoundError:
            self.skipTest("URDFファイルが見つからないため、エピソード完了テストをスキップ")
        except Exception as e:
            self.fail(f"エピソード完了テストに失敗: {e}")
    
    def test_debug_info_availability(self):
        """デバッグ情報の取得テスト"""
        if self.use_mock:
            self.skipTest("モックモードではデバッグ情報テストをスキップ")
        
        print("デバッグ情報取得テスト実行中...")
        
        try:
            self.env = BittleEnvironment(self.config)
            obs, _ = self.env.reset()
            
            # デバッグ情報の取得
            debug_info = self.env.get_debug_info()
            
            # デバッグ情報の構造確認
            self.assertIsInstance(debug_info, dict)
            self.assertIn('debug_info', debug_info)
            self.assertIn('episode_stats', debug_info)
            print("✓ デバッグ情報の構造確認成功")
            
            # エピソード統計の確認
            episode_stats = debug_info['episode_stats']
            self.assertIn('total_episodes', episode_stats)
            self.assertIn('current_episode_steps', episode_stats)
            print("✓ エピソード統計確認成功")
            
        except FileNotFoundError:
            self.skipTest("URDFファイルが見つからないため、デバッグ情報テストをスキップ")
        except Exception as e:
            self.fail(f"デバッグ情報取得テストに失敗: {e}")
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        print("エラーハンドリングテスト実行中...")
        
        # 不正な設定でのエラーテスト
        invalid_config = self.config.copy()
        invalid_config['robot']['urdf_path'] = 'nonexistent.urdf'
        
        with self.assertRaises((URDFLoadError, FileNotFoundError)):
            BittleEnvironment(invalid_config)
        print("✓ 不正なURDFパスのエラーハンドリング成功")
        
        # 設定必須項目欠如のテスト
        incomplete_config = {'environment': {}}
        with self.assertRaises(ConfigValidationError):
            BittleEnvironment(incomplete_config)
        print("✓ 不完全な設定のエラーハンドリング成功")


class TestEnvironmentIntegration(unittest.TestCase):
    """環境の統合テスト"""
    
    def setUp(self):
        """統合テストの初期化"""
        self.config = create_default_config()
        self.temp_dir = tempfile.mkdtemp()
        self.config['logging']['log_dir'] = os.path.join(self.temp_dir, 'logs')
        self.use_mock = os.environ.get('BITTLE_USE_MOCK', 'false').lower() == 'true'
    
    def tearDown(self):
        """統合テストの後処理"""
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except:
                pass
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_multiple_episodes(self):
        """複数エピソードの実行テスト"""
        if self.use_mock:
            self.skipTest("モックモードでは複数エピソードテストをスキップ")
        
        print("複数エピソード実行テスト開始...")
        
        try:
            self.env = BittleEnvironment(self.config)
            
            for episode in range(3):
                print(f"  エピソード {episode + 1} 実行中...")
                obs, info = self.env.reset()
                
                steps = 0
                while steps < 50:  # 短いエピソード
                    action = self.env.action_space.sample()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                print(f"  エピソード {episode + 1} 完了 (ステップ数: {steps})")
            
            print("✓ 複数エピソード実行テスト成功")
            
        except FileNotFoundError:
            self.skipTest("URDFファイルが見つからないため、複数エピソードテストをスキップ")
        except Exception as e:
            self.fail(f"複数エピソード実行テストに失敗: {e}")


def run_environment_tests():
    """環境テストの実行"""
    print("=== Bittle環境テスト開始 ===\n")
    
    # テストスイートの作成
    test_suite = unittest.TestSuite()
    
    # テストケースの追加
    test_suite.addTest(unittest.makeSuite(TestBittleEnvironment))
    test_suite.addTest(unittest.makeSuite(TestEnvironmentIntegration))
    
    # テストランナーの実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 結果の表示
    print(f"\n=== テスト結果 ===")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    print(f"スキップ: {len(result.skipped)}")
    
    if result.failures:
        print("\n=== 失敗したテスト ===")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n=== エラーが発生したテスト ===")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_environment_tests()
    exit(0 if success else 1)
