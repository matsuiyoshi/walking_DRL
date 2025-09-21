"""
設定検証ユーティリティ
設定の妥当性チェックとデバッグ支援
"""

import os
from typing import Dict, Any, List, Union
from pathlib import Path
import yaml
from .exceptions import ConfigValidationError
from .logger import get_logger

logger = get_logger("config_validator")


class ConfigValidator:
    """設定検証クラス"""
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
    
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """検証ルールの定義"""
        return {
            'environment': {
                'max_episode_steps': {
                    'type': int,
                    'min_value': 1,
                    'max_value': 10000,
                    'required': True
                },
                'control_frequency': {
                    'type': (int, float),
                    'min_value': 1,
                    'max_value': 1000,
                    'required': True
                },
                'physics_frequency': {
                    'type': (int, float),
                    'min_value': 1,
                    'max_value': 2000,
                    'required': True
                },
                'name': {
                    'type': str,
                    'required': True
                }
            },
            'robot': {
                'urdf_path': {
                    'type': str,
                    'required': True,
                    'file_exists': True
                },
                'initial_position': {
                    'type': list,
                    'length': 3,
                    'required': False
                },
                'initial_orientation': {
                    'type': list,
                    'length': 3,
                    'required': False
                },
                'joint_limits': {
                    'type': list,
                    'length': 2,
                    'required': False
                }
            },
            'training': {
                'algorithm': {
                    'type': str,
                    'allowed_values': ['PPO', 'SAC', 'TD3'],
                    'required': True
                },
                'total_timesteps': {
                    'type': int,
                    'min_value': 1000,
                    'max_value': 100000000,
                    'required': True
                },
                'learning_rate': {
                    'type': float,
                    'min_value': 1e-6,
                    'max_value': 1.0,
                    'required': True
                },
                'batch_size': {
                    'type': int,
                    'min_value': 1,
                    'max_value': 1024,
                    'required': True
                }
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        設定の妥当性を検証
        
        Args:
            config: 検証する設定辞書
            
        Returns:
            bool: 検証成功時True
            
        Raises:
            ConfigValidationError: 検証失敗時
        """
        logger.info("設定の検証を開始", {"config_keys": list(config.keys())})
        
        try:
            # 基本構造の検証
            self._validate_structure(config)
            
            # 各セクションの詳細検証
            for section, section_config in config.items():
                if section in self.validation_rules:
                    self._validate_section(section, section_config)
            
            # 依存関係の検証
            self._validate_dependencies(config)
            
            logger.info("設定の検証が成功しました")
            return True
            
        except ConfigValidationError as e:
            logger.error("設定の検証に失敗しました", exception=e)
            raise
        except Exception as e:
            logger.error("設定検証中に予期しないエラーが発生しました", exception=e)
            raise ConfigValidationError("unknown", "valid config", str(e))
    
    def _validate_structure(self, config: Dict[str, Any]):
        """基本構造の検証"""
        if not isinstance(config, dict):
            raise ConfigValidationError("root", "dict", type(config).__name__)
        
        # 必須セクションの確認
        required_sections = ['environment']
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(section, "required section", "missing")
    
    def _validate_section(self, section_name: str, section_config: Dict[str, Any]):
        """セクションの詳細検証"""
        logger.debug(f"セクション '{section_name}' の検証開始")
        
        rules = self.validation_rules[section_name]
        
        for key, rule in rules.items():
            if rule.get('required', False) and key not in section_config:
                raise ConfigValidationError(
                    f"{section_name}.{key}",
                    "required field",
                    "missing"
                )
            
            if key in section_config:
                self._validate_field(f"{section_name}.{key}", section_config[key], rule)
    
    def _validate_field(self, field_path: str, value: Any, rule: Dict[str, Any]):
        """フィールドの検証"""
        # 型チェック
        expected_type = rule.get('type')
        if expected_type and not isinstance(value, expected_type):
            raise ConfigValidationError(
                field_path,
                str(expected_type),
                type(value).__name__
            )
        
        # 範囲チェック
        if 'min_value' in rule and value < rule['min_value']:
            raise ConfigValidationError(
                field_path,
                f">= {rule['min_value']}",
                str(value)
            )
        
        if 'max_value' in rule and value > rule['max_value']:
            raise ConfigValidationError(
                field_path,
                f"<= {rule['max_value']}",
                str(value)
            )
        
        # 許可値チェック
        if 'allowed_values' in rule and value not in rule['allowed_values']:
            raise ConfigValidationError(
                field_path,
                f"one of {rule['allowed_values']}",
                str(value)
            )
        
        # リスト長チェック
        if 'length' in rule and len(value) != rule['length']:
            raise ConfigValidationError(
                field_path,
                f"length {rule['length']}",
                f"length {len(value)}"
            )
        
        # ファイル存在チェック（実験時は警告のみ）
        if rule.get('file_exists') and not os.path.exists(value):
            logger.warning(f"ファイルが見つかりません: {value}")
            # 学習実行時のみチェックする場合はコメントアウト
            # raise ConfigValidationError(
            #     field_path,
            #     "existing file", 
            #     f"file not found: {value}"
            # )
    
    def _validate_dependencies(self, config: Dict[str, Any]):
        """依存関係の検証"""
        # 制御周波数 <= 物理周波数
        env_config = config.get('environment', {})
        control_freq = env_config.get('control_frequency')
        physics_freq = env_config.get('physics_frequency')
        
        if control_freq and physics_freq and control_freq > physics_freq:
            raise ConfigValidationError(
                "environment.control_frequency",
                f"<= physics_frequency ({physics_freq})",
                str(control_freq)
            )
        
        # URDFファイルのパス解決
        robot_config = config.get('robot', {})
        if 'urdf_path' in robot_config:
            urdf_path = robot_config['urdf_path']
            if not os.path.isabs(urdf_path):
                # 相対パスの場合、assetsディレクトリからの相対パスとして解決
                base_path = Path(__file__).parent.parent.parent / 'assets' / 'bittle-urdf'
                full_path = base_path / urdf_path
                if not full_path.exists():
                    # URDFファイルが見つからない場合は警告のみ（テスト時は存在しない場合がある）
                    logger.warning(f"URDFファイルが見つかりません: {full_path}")
                    logger.info("学習実行時にはURDFファイルが必要です")
                    # 実行時は例外を投げずに警告のみとする


def validate_config(config: Dict[str, Any]) -> bool:
    """
    設定検証の便利関数
    
    Args:
        config: 検証する設定辞書
        
    Returns:
        bool: 検証成功時True
    """
    validator = ConfigValidator()
    return validator.validate_config(config)


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """
    設定ファイルの読み込みと検証
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        Dict[str, Any]: 検証済み設定辞書
        
    Raises:
        ConfigValidationError: 検証失敗時
        FileNotFoundError: ファイルが見つからない場合
    """
    logger.info("設定ファイルの読み込み", {"config_path": config_path})
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.debug("設定ファイルの読み込み完了", {"config": config})
        
        # 検証実行
        validate_config(config)
        
        return config
        
    except yaml.YAMLError as e:
        logger.error("YAML解析エラー", exception=e)
        raise ConfigValidationError("yaml_parse", "valid YAML", str(e))
    except Exception as e:
        logger.error("設定ファイル読み込みエラー", exception=e)
        raise


def create_default_config() -> Dict[str, Any]:
    """
    デフォルト設定の作成
    
    Returns:
        Dict[str, Any]: デフォルト設定辞書
    """
    return {
        'environment': {
            'name': "BittleWalking-v0",
            'max_episode_steps': 500,
            'control_frequency': 50,
            'physics_frequency': 240
        },
        'robot': {
            'urdf_path': "bittle.urdf",
            'initial_position': [0.0, 0.0, 0.1],
            'initial_orientation': [0.0, 0.0, 0.0],
            'joint_limits': [-1.57, 1.57],
            'max_torque': 10.0
        },
        'rewards': {
            'forward_velocity_weight': 10.0,
            'survival_reward': 1.0,
            'fall_penalty': -100.0,
            'energy_efficiency_weight': -0.01
        },
        'termination': {
            'min_height': 0.05,
            'max_roll': 45.0,
            'max_pitch': 45.0
        },
        'training': {
            'algorithm': "PPO",
            'total_timesteps': 1000000,
            'learning_rate': 0.0003,
            'batch_size': 64,
            'n_envs': 1,
            'n_steps': 2048,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5
        },
        'save': {
            'frequency': 100000,
            'model_path': "./models",
            'checkpoint_path': "./models/checkpoints",
            'vec_normalize_path': "./models/vec_normalize.pkl"
        },
        'evaluation': {
            'frequency': 50000,
            'n_eval_episodes': 5,
            'deterministic': True
        },
        'logging': {
            'log_dir': "./logs",
            'tensorboard': True,
            'verbose': 1,
            'debug_level': "INFO"
        }
    }
