#!/usr/bin/env python3
"""
Bittle四足歩行ロボット - 最小クイックテストスクリプト
依存関係なしで実行可能な基本テスト
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """基本的なインポートテスト（依存関係なし）"""
    print("=== 基本インポートテスト ===")
    
    try:
        print("ユーティリティクラスのインポート...")
        from src.utils.exceptions import EnvironmentError
        from src.utils.logger import setup_logger
        print("✓ ユーティリティクラスのインポート成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        return False

def test_logger_minimal():
    """ロガーの最小テスト"""
    print("\n=== ロガー最小テスト ===")
    
    try:
        from src.utils.logger import get_logger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = get_logger("test_logger", temp_dir, "DEBUG")
            
            logger.info("テストメッセージ: INFO")
            logger.debug("テストメッセージ: DEBUG")
            
            # ログファイルの確認
            log_files = list(Path(temp_dir).glob("*.log"))
            if log_files:
                print(f"✓ ログファイル作成成功: {len(log_files)}個")
                return True
            else:
                print("✗ ログファイルが作成されませんでした")
                return False
        
    except Exception as e:
        print(f"✗ ロガーテストエラー: {e}")
        return False

def test_config_minimal():
    """設定の最小テスト"""
    print("\n=== 設定最小テスト ===")
    
    try:
        from src.utils.config_validator import create_default_config
        
        # デフォルト設定の作成
        config = create_default_config()
        
        # 基本的な構造確認
        required_sections = ['environment', 'robot', 'training']
        for section in required_sections:
            if section not in config:
                print(f"✗ 必須セクション不足: {section}")
                return False
        
        print("✓ 設定構造確認成功")
        
        # URDFパスチェック（警告レベル）
        if 'urdf_path' in config.get('robot', {}):
            urdf_path = config['robot']['urdf_path']
            assets_path = Path(__file__).parent / 'assets' / 'bittle-urdf' / urdf_path
            if not assets_path.exists():
                print(f"⚠ URDFファイルが見つかりません: {assets_path}")
                print("  学習実行時にはURDFファイルが必要です")
            else:
                print("✓ URDFファイル確認成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 設定テストエラー: {e}")
        return False

def test_project_structure():
    """プロジェクト構造の確認"""
    print("\n=== プロジェクト構造確認 ===")
    
    required_dirs = [
        'src',
        'src/utils',
        'configs',
        'tests',
        'assets',
        'assets/bittle-urdf'
    ]
    
    required_files = [
        'src/__init__.py',
        'src/utils/__init__.py',
        'src/utils/exceptions.py',
        'src/utils/logger.py',
        'src/utils/config_validator.py',
        'configs/default.yaml',
        'requirements.txt',
        'README.md'
    ]
    
    project_root = Path(__file__).parent
    
    # ディレクトリの確認
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✓ {dir_path}")
    
    # ファイルの確認
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_dirs:
        print(f"✗ 不足ディレクトリ: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"✗ 不足ファイル: {missing_files}")
        return False
    
    print("✓ プロジェクト構造確認成功")
    return True

def check_python_version():
    """Python版本の確認"""
    print("\n=== Python版本確認 ===")
    
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8以上が必要です")
        return False
    
    print("✓ Python版本確認成功")
    return True

def main():
    """メイン実行関数"""
    print("Bittle四足歩行ロボット - 最小クイックテスト開始")
    print("=" * 50)
    
    tests = [
        ("Python版本確認", check_python_version),
        ("プロジェクト構造確認", test_project_structure),
        ("基本インポートテスト", test_basic_imports),
        ("ロガー最小テスト", test_logger_minimal),
        ("設定最小テスト", test_config_minimal),
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
        print("✓ 全ての最小テストが成功しました！")
        print("\n次のステップ:")
        print("1. 依存関係のインストール: pip install -r requirements.txt")
        print("2. フルテストの実行: python run_quick_test.py")
        print("3. 学習の開始: python -m src.training")
        return True
    else:
        print("✗ いくつかのテストが失敗しました。")
        print("プロジェクト構造や基本設定を確認してください。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
