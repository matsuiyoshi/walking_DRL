# Bittle四足歩行ロボット深層強化学習プロジェクト

Bittle四足歩行ロボットをPyBulletシミュレーション環境で深層強化学習により基本的な前進歩行を学習させるプロジェクトです。

## 🎯 プロジェクトの特徴

- **シンプル**: 基本的な前進歩行のみに焦点を当てた実装
- **デバッグ重視**: 詳細なログとエラーハンドリング機能
- **モジュラー設計**: 再利用可能なコンポーネント構成
- **Docker対応**: 環境構築の簡素化とポータビリティ
- **段階的実装**: 環境→学習→評価の順で確実に構築
- **厳密な仕様**: 実装可能な詳細仕様に基づく開発
- **運動学的計算**: 関節角度から足先位置を正確に計算（NEW!）

## 📁 プロジェクト構造

```
walking_DRL/
├── src/                          # ソースコード
│   ├── __init__.py
│   ├── environment.py           # PyBullet環境クラス
│   ├── training.py              # 学習スクリプト
│   ├── evaluation.py            # 評価スクリプト
│   └── utils/                   # ユーティリティ
│       ├── __init__.py
│       ├── exceptions.py        # カスタム例外
│       ├── logger.py           # ログ管理
│       └── config_validator.py  # 設定検証
├── configs/                     # 設定ファイル
│   ├── default.yaml            # デフォルト設定
│   ├── training.yaml           # 学習設定
│   └── evaluation.yaml         # 評価設定
├── tests/                      # テストファイル
│   ├── __init__.py
│   ├── test_environment.py     # 環境テスト
│   ├── test_training.py        # 学習テスト
│   └── test_utils.py           # ユーティリティテスト
├── assets/                     # リソースファイル
│   └── bittle-urdf/           # ロボットモデル
├── models/                     # 学習済みモデル
│   └── checkpoints/           # チェックポイント
├── logs/                      # ログファイル
│   └── tensorboard/           # TensorBoardログ
├── evaluation_results/        # 評価結果
├── notebooks/                 # Jupyter Notebook
│   ├── 01_test_environment.ipynb
│   └── 02_training_results.ipynb
├── run_quick_test.py          # クイックテスト
├── run_quick_test_minimal.py  # 最小限クイックテスト
├── requirements.txt           # 依存関係
├── requirements-minimal.txt   # 最小限依存関係
├── Dockerfile                 # Docker設定
├── docker-entrypoint.sh      # Dockerエントリーポイント
└── README.md                  # このファイル
```

## 🚀 クイックスタート

### 1. 依存関係の確認

```bash
# 基本的なクイックテストの実行
python run_quick_test.py

# 最小限の依存関係でのテスト
python run_quick_test_minimal.py
```

### 2. Docker環境での実行（推奨）

```bash
# Dockerイメージのビルド
docker build -t bittle-walking .

# 学習の実行
docker run --gpus all -v $(pwd):/app bittle-walking train

# 評価の実行
docker run --gpus all -v $(pwd):/app bittle-walking evaluate models/best_model/best_model.zip

# インタラクティブ評価（GUI）
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/app bittle-walking interactive models/final_model.zip
```

### 3. ローカル環境での実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# または最小限の依存関係
pip install -r requirements-minimal.txt

# 環境テストの実行
python tests/test_environment.py

# 学習の実行
python -m src.training

# 評価の実行
python -m src.evaluation models/final_model.zip
```

## 🔧 設定

### 環境設定 (`configs/default.yaml`)

```yaml
# 環境設定
environment:
  name: "BittleWalking-v0"
  max_episode_steps: 500
  control_frequency: 50
  physics_frequency: 240
  gravity: -9.81
  time_step: 0.004  # 1/240
  
# ロボット設定
robot:
  urdf_path: "assets/bittle-urdf/bittle.urdf"
  initial_position: [0.0, 0.0, 0.1]
  initial_orientation: [0.0, 0.0, 0.0]
  joint_limits: [-1.57, 1.57]
  max_torque: 10.0
  
# 報酬設定
rewards:
  forward_velocity_weight: 10.0
  survival_reward: 1.0
  fall_penalty: -100.0
  energy_efficiency_weight: -0.01
  
# 終了条件
termination:
  max_height: 0.05
  max_roll: 45.0  # 度
  max_pitch: 45.0  # 度
```

### 学習設定 (`configs/training.yaml`)

```yaml
# 学習設定
training:
  algorithm: "PPO"
  total_timesteps: 1000000
  n_envs: 4
  n_steps: 2048
  learning_rate: 0.0003
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  
# 保存設定
save:
  frequency: 100000
  model_path: "./models"
  checkpoint_path: "./models/checkpoints"
  vec_normalize_path: "./models/vec_normalize.pkl"
  
# 評価設定
evaluation:
  frequency: 50000
  n_eval_episodes: 5
  deterministic: true
```

## 📊 学習・評価

### 学習の実行

```bash
# デフォルト設定での学習
python -m src.training

# カスタム設定での学習
python -m src.training --config configs/custom.yaml

# デバッグモードでの学習
python -m src.training --debug
```

**🔄 Best Modelの自動バックアップ機能**

新しい学習を開始すると、前回の`best_model.zip`が自動的に`models/best_model_archive/`にバックアップされます。報酬設定を変更して再学習しても、過去の優秀なモデルは失われません。

```bash
# 学習実行（自動でバックアップ）
python -m src.training configs/production.yaml

# バックアップされたモデルの確認
ls -lh models/best_model_archive/
# → best_model_20251006_190411.zip
# → best_model_20251007_153020.zip

# 過去のモデルを評価
python -m src.evaluation models/best_model_archive/best_model_20251006_190411.zip
```

詳細は[BEST_MODEL_BACKUP_GUIDE.md](BEST_MODEL_BACKUP_GUIDE.md)を参照してください。

### 評価の実行

```bash
# 基本的な評価
python -m src.evaluation models/best_model.zip

# 詳細評価（可視化付き）
python -m src.evaluation models/best_model.zip --render --episodes 20

# インタラクティブ評価
python -m src.evaluation models/best_model.zip --interactive
```

### 運動学的足先接地報酬（NEW!）

関節角度から三角関数を使って足先位置を計算し、地面との距離を把握する運動学的計算を実装しました。

```bash
# 運動学的計算を使用した学習
python -m src.training --config configs/production.yaml

# 設定ファイルで切り替え可能
# configs/production.yaml: use_kinematic_contact: true  (運動学的計算)
# configs/debug.yaml: use_kinematic_contact: false (物理エンジンの接触検知)
```

**主な機能:**
- 関節角度から足先位置を正確に計算
- 左右対称を考慮（モーターの回転方向）
- 物理シミュレーションに依存しない接地判定
- 既存の実装との完全な互換性

詳細は以下のドキュメントを参照：
- [KINEMATIC_FOOT_CONTACT_IMPLEMENTATION.md](KINEMATIC_FOOT_CONTACT_IMPLEMENTATION.md): 実装の詳細ガイド
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): 実装完了報告

## 🐛 デバッグ機能

### ログファイル

- `logs/environment_YYYYMMDD_HHMMSS.log`: 環境関連のログ
- `logs/bittle_trainer_YYYYMMDD_HHMMSS.log`: 学習関連のログ
- `logs/tensorboard/`: TensorBoardログ

### デバッグ情報の取得

```python
from src.environment import BittleEnvironment

env = BittleEnvironment(config)
debug_info = env.get_debug_info()
print(debug_info)
```

### エラーハンドリング

```python
from src.utils.exceptions import EnvironmentError, URDFLoadError

try:
    env = BittleEnvironment(config)
except URDFLoadError as e:
    print(f"URDF読み込みエラー: {e}")
    print(f"詳細: {e.details}")
```

## 🧪 テスト

### 環境テストの実行

```bash
# 全ての環境テスト
python tests/test_environment.py

# モック環境でのテスト（PyBulletなし）
BITTLE_USE_MOCK=true python tests/test_environment.py

# Pytestでの実行
pytest tests/ -v
```

### テストカバレッジ

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 パフォーマンス監視

### TensorBoard

```bash
tensorboard --logdir=logs/tensorboard
```

### 評価結果の可視化

```python
from src.evaluation import BittleEvaluator

evaluator = BittleEvaluator("models/best_model.zip")
results = evaluator.evaluate_model(num_episodes=10)

# 結果は evaluation_results/ に保存
```

## 🔧 トラブルシューティング

### よくある問題

1. **URDFファイルが見つからない**
   ```
   FileNotFoundError: URDF file not found
   ```
   → `assets/bittle-urdf/bittle.urdf` の存在を確認
   → プロジェクトルートから実行しているか確認

2. **PyBullet初期化エラー**
   ```
   PhysicsInitializationError: Physics engine initialization failed
   ```
   → Dockerを使用するか、X11設定を確認
   → `run_quick_test.py` で基本動作を確認

3. **メモリ不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   → `batch_size` を減らすか、`n_envs` を調整
   → `requirements-minimal.txt` を使用して軽量化

4. **学習が収束しない**
   - 報酬関数の重みを調整（`configs/default.yaml`）
   - `learning_rate` を変更（`configs/training.yaml`）
   - エピソード長を確認
   - TensorBoardで学習曲線を確認

5. **インポートエラー**
   ```
   ImportError: No module named 'src'
   ```
   → プロジェクトルートから実行
   → `python -m src.training` の形式で実行

6. **設定検証エラー**
   ```
   ConfigValidationError: Control frequency must be <= physics frequency
   ```
   → `configs/default.yaml` の設定値を確認
   → `control_frequency` ≤ `physics_frequency` を確認

### ログレベルの調整

```yaml
logging:
  debug_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

### 設定の検証

```python
from src.utils.config_validator import validate_config, create_default_config

# デフォルト設定の作成
config = create_default_config()

# 設定の検証
try:
    validate_config(config)
    print("✓ 設定が正常です")
except ConfigValidationError as e:
    print(f"✗ 設定エラー: {e}")
```

### デバッグ情報の取得

```python
from src.environment import BittleEnvironment

env = BittleEnvironment(config)
debug_info = env.get_debug_info()
print(debug_info)
```

### クイックテストの実行

```bash
# 基本的な動作確認
python run_quick_test.py

# 最小限の依存関係での確認
python run_quick_test_minimal.py

# 特定のテストのみ実行
python -c "from run_quick_test import test_imports; test_imports()"
```

## 📚 開発者向け情報

### 環境の拡張

```python
class CustomBittleEnvironment(BittleEnvironment):
    def _calculate_reward_detailed(self, action):
        # カスタム報酬関数の実装
        return reward, reward_breakdown
```

### カスタムコールバック

```python
from src.training import DebugCallback

class CustomCallback(DebugCallback):
    def _on_step(self):
        # カスタム処理
        return super()._on_step()
```

### 新しいテストの追加

```python
class TestCustomFeature(unittest.TestCase):
    def test_feature(self):
        # テストの実装
        pass
```

## 🏆 成功指標

### 学習目標

```python
SUCCESS_CRITERIA = {
    'average_reward': {
        'target': 100.0,
        'tolerance': 10.0,
        'description': '平均報酬'
    },
    'forward_velocity': {
        'target': 0.1,  # m/s
        'tolerance': 0.02,
        'description': '前進速度'
    },
    'survival_rate': {
        'target': 0.8,
        'tolerance': 0.1,
        'description': '生存率'
    },
    'stability': {
        'target': 0.9,
        'tolerance': 0.05,
        'description': '安定性指標'
    }
}
```

### 評価方法

- **平均報酬**: 100以上（±10の許容範囲）
- **前進速度**: 0.1 m/s以上（±0.02 m/sの許容範囲）
- **生存率**: 80%以上（±10%の許容範囲）
- **安定性**: 90%以上（±5%の許容範囲）

### 技術的制約

```python
RESOURCE_REQUIREMENTS = {
    'gpu': {
        'required': True,
        'memory': '8GB+',
        'cuda_version': '11.8+'
    },
    'cpu': {
        'cores': 4,
        'memory': '16GB+'
    },
    'storage': {
        'models': '2GB',
        'logs': '1GB',
        'total': '5GB+'
    }
}
```

### 実行時間

- **環境セットアップ**: 5-10分
- **学習**: 2-4時間
- **評価**: 10-20分
- **合計**: 3-5時間

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 🚀 実装順序

### Phase 1: 環境実装 (1-2日)
1. `src/environment.py` の実装
2. 基本的なPyBullet環境の構築
3. 状態・行動空間の定義
4. 基本報酬関数の実装
5. 環境テストの作成

### Phase 2: 学習実装 (2-3日)
1. `src/training.py` の実装
2. PPOエージェントの統合
3. 学習ループの構築
4. ログ・可視化の追加
5. 学習テストの作成

### Phase 3: 評価・可視化 (1-2日)
1. `src/evaluation.py` の実装
2. Jupyter Notebookの作成
3. 結果可視化の実装
4. 性能評価の実行

## 📊 状態・行動空間の詳細

### 状態空間 (22次元)
```python
OBSERVATION_SPACE_SPEC = {
    'joint_angles': {
        'dimension': 8,
        'range': [-np.pi, np.pi],
        'description': '4脚 × 2関節の角度'
    },
    'robot_orientation': {
        'dimension': 3,
        'range': [-np.pi, np.pi],
        'description': 'roll, pitch, yaw'
    },
    'robot_velocity': {
        'dimension': 3,
        'range': [-10.0, 10.0],
        'description': 'vx, vy, vz'
    },
    'last_action': {
        'dimension': 8,
        'range': [-np.pi, np.pi],
        'description': '前回のアクション'
    }
}
```

### 行動空間 (8次元)
```python
ACTION_SPACE_SPEC = {
    'joint_target_angles': {
        'dimension': 8,
        'range': [-1.57, 1.57],  # ±90度
        'description': '4脚 × 2関節の目標角度'
    }
}
```

## 📞 サポート

問題が発生した場合は、以下を確認してください：

1. [トラブルシューティング](#🔧-トラブルシューティング)
2. ログファイル (`logs/`)
3. デバッグ情報の確認
4. クイックテストの実行

---

**Happy Robot Walking! 🤖🚶‍♂️**
