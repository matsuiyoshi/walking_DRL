"""
ログ管理ユーティリティ
デバッグとモニタリングを支援するログシステム
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class DebugLogger:
    """デバッグ用ロガークラス"""
    
    def __init__(self, name: str, log_dir: str = "./logs", debug_level: str = "INFO"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # メインロガーの設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, debug_level.upper()))
        
        # 既存のハンドラーを削除（重複防止）
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # ファイルハンドラーの設定
        self._setup_file_handler()
        
        # コンソールハンドラーの設定
        self._setup_console_handler()
        
        # セッション開始ログ
        self.info(f"=== ログセッション開始: {datetime.now()} ===")
    
    def _setup_file_handler(self):
        """ファイルハンドラーの設定"""
        # 日付付きログファイル
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 詳細フォーマット
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
        
        # 最新ログファイルのシンボリックリンク作成
        latest_log = self.log_dir / f"{self.name}_latest.log"
        if latest_log.exists():
            latest_log.unlink()
        latest_log.symlink_to(log_file.name)
    
    def _setup_console_handler(self):
        """コンソールハンドラーの設定"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # シンプルフォーマット
        console_formatter = logging.Formatter(
            '%(levelname)s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """デバッグレベルログ"""
        full_message = message
        if extra_data:
            full_message += f" | Data: {json.dumps(extra_data, ensure_ascii=False)}"
        self.logger.debug(full_message)
    
    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """情報レベルログ"""
        full_message = message
        if extra_data:
            full_message += f" | Data: {json.dumps(extra_data, ensure_ascii=False)}"
        self.logger.info(full_message)
    
    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """警告レベルログ"""
        full_message = message
        if extra_data:
            full_message += f" | Data: {json.dumps(extra_data, ensure_ascii=False)}"
        self.logger.warning(full_message)
    
    def error(self, message: str, exception: Exception = None, extra_data: Dict[str, Any] = None):
        """エラーレベルログ"""
        full_message = message
        if exception:
            full_message += f" | Exception: {type(exception).__name__}: {str(exception)}"
        if extra_data:
            full_message += f" | Data: {json.dumps(extra_data, ensure_ascii=False)}"
        self.logger.error(full_message)
        
        if exception:
            self.logger.debug("Exception traceback:", exc_info=True)
    
    def critical(self, message: str, exception: Exception = None, extra_data: Dict[str, Any] = None):
        """重大エラーレベルログ"""
        full_message = message
        if exception:
            full_message += f" | Exception: {type(exception).__name__}: {str(exception)}"
        if extra_data:
            full_message += f" | Data: {json.dumps(extra_data, ensure_ascii=False)}"
        self.logger.critical(full_message)
        
        if exception:
            self.logger.debug("Exception traceback:", exc_info=True)
    
    def log_function_entry(self, func_name: str, args: tuple = None, kwargs: dict = None):
        """関数の開始ログ"""
        args_str = f"args={args}" if args else ""
        kwargs_str = f"kwargs={kwargs}" if kwargs else ""
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        self.debug(f">>> {func_name}({params})")
    
    def log_function_exit(self, func_name: str, result: Any = None, execution_time: float = None):
        """関数の終了ログ"""
        time_str = f"({execution_time:.3f}s)" if execution_time else ""
        result_str = f"-> {result}" if result is not None else ""
        self.debug(f"<<< {func_name}{time_str} {result_str}")
    
    def log_state_change(self, component: str, old_state: Any, new_state: Any):
        """状態変化のログ"""
        self.info(f"State change in {component}: {old_state} -> {new_state}")
    
    def log_performance(self, operation: str, duration: float, extra_metrics: Dict[str, Any] = None):
        """パフォーマンスログ"""
        message = f"Performance | {operation}: {duration:.3f}s"
        if extra_metrics:
            message += f" | Metrics: {json.dumps(extra_metrics, ensure_ascii=False)}"
        self.info(message)


def setup_logger(name: str, log_dir: str = "./logs", debug_level: str = "INFO") -> DebugLogger:
    """
    デバッグロガーの設定
    
    Args:
        name: ロガー名
        log_dir: ログディレクトリ
        debug_level: ログレベル
        
    Returns:
        DebugLogger: 設定されたロガー
    """
    return DebugLogger(name, log_dir, debug_level)


# グローバルロガーインスタンス
_loggers = {}

def get_logger(name: str, log_dir: str = "./logs", debug_level: str = "INFO") -> DebugLogger:
    """
    ロガーインスタンスの取得（シングルトンパターン）
    
    Args:
        name: ロガー名
        log_dir: ログディレクトリ
        debug_level: ログレベル
        
    Returns:
        DebugLogger: ロガーインスタンス
    """
    if name not in _loggers:
        _loggers[name] = setup_logger(name, log_dir, debug_level)
    return _loggers[name]
