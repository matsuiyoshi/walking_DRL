#!/usr/bin/env python3
"""
Bittle四足歩行ロボット - クイックテストスクリプト
実装の基本動作を迅速に確認するためのスクリプト
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """基本的なインポートテスト"""
    print("=== インポートテスト ===")
    
    try:
        print("ユーティリティクラスのインポート...")
        from src.utils.exceptions import EnvironmentError
        from src.utils.logger import setup_logger
        from src.utils.config_validator import create_default_config, validate_config
        print("✓ ユーティリティクラスのインポート成功")
        
        print("設定作成・検証テスト...")
        config = create_default_config()
        validate_config(config)
        print("✓ 設定作成・検証成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        return False

def test_logger():
    """ロガーのテスト"""
    print("\n=== ロガーテスト ===")
    
    try:
        from src.utils.logger import get_logger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = get_logger("test_logger", temp_dir, "DEBUG")
            
            logger.info("テストメッセージ: INFO")
            logger.debug("テストメッセージ: DEBUG")
            logger.warning("テストメッセージ: WARNING")
            
            # ログファイルの確認
            log_files = list(Path(temp_dir).glob("*.log"))
            if log_files:
                print(f"✓ ログファイル作成成功: {len(log_files)}個")
            else:
                print("✗ ログファイルが作成されませんでした")
                return False
        
        print("✓ ロガーテスト成功")
        return True
        
    except Exception as e:
        print(f"✗ ロガーテストエラー: {e}")
        return False

def test_config_validator():
    """設定検証のテスト"""
    print("\n=== 設定検証テスト ===")
    
    try:
        from src.utils.config_validator import validate_config, ConfigValidationError
        from src.utils.config_validator import create_default_config
        
        # 正常な設定のテスト
        config = create_default_config()
        validate_config(config)
        print("✓ 正常な設定の検証成功")
        
        # 不正な設定のテスト
        invalid_config = config.copy()
        invalid_config['environment']['control_frequency'] = 1000
        invalid_config['environment']['physics_frequency'] = 100
        
        try:
            validate_config(invalid_config)
            print("✗ 不正な設定が検証を通過してしまいました")
            return False
        except ConfigValidationError:
            print("✓ 不正な設定の検証成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 設定検証テストエラー: {e}")
        return False

def test_environment_creation():
    """環境作成のテスト（PyBulletなし）"""
    print("\n=== 環境作成テスト（簡易版）===")
    
    try:
        from src.utils.config_validator import create_default_config
        
        # 設定の準備
        config = create_default_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config['logging']['log_dir'] = temp_dir
            
            # 環境クラスのインポートテスト
            from src.environment import BittleEnvironment
            print("✓ 環境クラスのインポート成功")
            
            # 注意: 実際の環境作成はURDFファイルが必要なためスキップ
            print("✓ 環境クラスの基本構造確認成功")
            print("  （実際の環境作成はURDFファイルが必要なため、統合テストで実行）")
        
        return True
        
    except Exception as e:
        print(f"✗ 環境作成テストエラー: {e}")
        return False

def test_training_module():
    """学習モジュールのテスト"""
    print("\n=== 学習モジュールテスト ===")
    
    try:
        # 学習モジュールのインポート
        from src.training import BittleTrainer
        print("✓ 学習モジュールのインポート成功")
        
        # トレーナークラスの基本チェック
        config_path = None  # デフォルト設定を使用
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 注意: 実際のトレーナー初期化はPyBulletが必要なためスキップ
            print("✓ 学習モジュールの基本構造確認成功")
            print("  （実際のトレーナー初期化はPyBulletが必要なため、統合テストで実行）")
        
        return True
        
    except Exception as e:
        print(f"✗ 学習モジュールテストエラー: {e}")
        return False

def test_evaluation_module():
    """評価モジュールのテスト"""
    print("\n=== 評価モジュールテスト ===")
    
    try:
        # 評価モジュールのインポート
        from src.evaluation import BittleEvaluator
        print("✓ 評価モジュールのインポート成功")
        
        print("✓ 評価モジュールの基本構造確認成功")
        print("  （実際の評価器初期化は学習済みモデルが必要なため、統合テストで実行）")
        
        return True
        
    except Exception as e:
        print(f"✗ 評価モジュールテストエラー: {e}")
        return False

def check_dependencies():
    """依存関係の確認"""
    print("\n=== 依存関係確認 ===")
    
    dependencies = [
        'numpy',
        'gymnasium', 
        'pybullet',
        'stable_baselines3',
        'yaml',
        'matplotlib'
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            if dep == 'yaml':
                import yaml
            else:
                __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} (未インストール)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n未インストールの依存関係: {missing_deps}")
        print("pip install -r requirements.txt を実行してください")
        return False
    
    print("✓ 全ての依存関係が確認されました")
    return True

def main():
    """メイン実行関数"""
    print("Bittle四足歩行ロボット - クイックテスト開始")
    print("=" * 50)
    
    tests = [
        ("依存関係確認", check_dependencies),
        ("インポートテスト", test_imports),
        ("ロガーテスト", test_logger),
        ("設定検証テスト", test_config_validator),
        ("環境作成テスト", test_environment_creation),
        ("学習モジュールテスト", test_training_module),
        ("評価モジュールテスト", test_evaluation_module),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: 予期しないエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("=== テスト結果サマリー ===")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n成功: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✓ 全てのクイックテストが成功しました！")
        print("\n次のステップ:")
        print("1. 統合テストの実行: python tests/test_environment.py")
        print("2. 学習の開始: python -m src.training")
        return True
    else:
        print("✗ いくつかのテストが失敗しました。")
        print("エラーメッセージを確認して問題を修正してください。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
