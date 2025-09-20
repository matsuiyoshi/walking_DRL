"""
カスタム例外クラス
デバッグとエラーハンドリングを支援する例外定義
"""

class EnvironmentError(Exception):
    """環境関連のエラー"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self):
        base_msg = super().__str__()
        if self.error_code:
            base_msg += f" [Error Code: {self.error_code}]"
        if self.details:
            base_msg += f" [Details: {self.details}]"
        return base_msg


class URDFLoadError(EnvironmentError):
    """URDF読み込みエラー"""
    def __init__(self, urdf_path: str, message: str = None):
        self.urdf_path = urdf_path
        message = message or f"URDF file could not be loaded: {urdf_path}"
        super().__init__(
            message, 
            error_code="URDF_LOAD_ERROR",
            details={"urdf_path": urdf_path}
        )


class PhysicsInitializationError(EnvironmentError):
    """物理エンジン初期化エラー"""
    def __init__(self, physics_client: int = None, message: str = None):
        self.physics_client = physics_client
        message = message or "Physics engine initialization failed"
        super().__init__(
            message,
            error_code="PHYSICS_INIT_ERROR", 
            details={"physics_client": physics_client}
        )


class ConfigValidationError(Exception):
    """設定検証エラー"""
    def __init__(self, config_key: str, expected_value: str = None, actual_value: str = None):
        self.config_key = config_key
        self.expected_value = expected_value
        self.actual_value = actual_value
        
        message = f"Configuration validation failed for key: {config_key}"
        if expected_value and actual_value:
            message += f" (expected: {expected_value}, got: {actual_value})"
            
        super().__init__(message)


class ModelLoadError(Exception):
    """モデル読み込みエラー"""
    def __init__(self, model_path: str, message: str = None):
        self.model_path = model_path
        message = message or f"Model could not be loaded: {model_path}"
        super().__init__(message)


class RobotStateError(EnvironmentError):
    """ロボット状態エラー"""
    def __init__(self, robot_id: int, state_info: dict = None, message: str = None):
        self.robot_id = robot_id
        self.state_info = state_info or {}
        message = message or f"Robot state error for robot_id: {robot_id}"
        super().__init__(
            message,
            error_code="ROBOT_STATE_ERROR",
            details={"robot_id": robot_id, "state_info": state_info}
        )


class ActionSpaceError(EnvironmentError):
    """アクション空間エラー"""
    def __init__(self, action, action_space_info: dict = None, message: str = None):
        self.action = action
        self.action_space_info = action_space_info or {}
        message = message or f"Invalid action: {action}"
        super().__init__(
            message,
            error_code="ACTION_SPACE_ERROR",
            details={"action": str(action), "action_space_info": action_space_info}
        )
