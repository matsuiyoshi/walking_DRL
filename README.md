# Bittle四足歩行ロボット深層強化学習プロジェクト

Bittle四足歩行ロボットをPyBulletシミュレーション環境で深層強化学習により基本的な前進歩行を学習させるプロジェクトです。

## 🎯 プロジェクトの特徴

- **シンプル**: 基本的な前進歩行のみに焦点
- **デバッグ重視**: 詳細なログとエラーハンドリング
- **モジュラー設計**: 再利用可能なコンポーネント
- **Docker対応**: 環境構築の簡素化

## 📁 プロジェクト構造

```
walking_DRL/
├── src/                          # ソースコード
│   ├── environment.py           # PyBullet環境クラス
│   ├── training.py              # 学習スクリプト
│   ├── evaluation.py            # 評価スクリプト
│   └── utils/                   # ユーティリティ
│       ├── exceptions.py        # カスタム例外
│       ├── logger.py           # ログ管理
│       └── config_validator.py  # 設定検証
├── configs/                     # 設定ファイル
│   └── default.yaml            # デフォルト設定
├── tests/                      # テストファイル
│   └── test_environment.py     # 環境テスト
├── assets/                     # リソースファイル
│   └── bittle-urdf/           # ロボットモデル
├── models/                     # 学習済みモデル
├── logs/                      # ログファイル
├── evaluation_results/        # 評価結果
└── notebooks/                 # Jupyter Notebook
```

## 🚀 クイックスタート

### 1. 依存関係の確認

```bash
# クイックテストの実行
python run_quick_test.py
```

### 2. Docker環境での実行（推奨）

```bash
# Dockerイメージのビルド
docker build -t bittle-walking .

# 学習の実行
docker run --gpus all -v $(pwd):/app bittle-walking train

# 評価の実行
docker run --gpus all -v $(pwd):/app bittle-walking evaluate models/final_model.zip

# インタラクティブ評価（GUI）
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/app bittle-walking interactive models/final_model.zip
```

### 3. ローカル環境での実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

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
environment:
  name: "BittleWalking-v0"
  max_episode_steps: 500
  control_frequency: 50
  physics_frequency: 240

training:
  algorithm: "PPO"
  total_timesteps: 1000000
  learning_rate: 0.0003
  batch_size: 64

rewards:
  forward_velocity_weight: 10.0
  survival_reward: 1.0
  fall_penalty: -100.0
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

### 評価の実行

```bash
# 基本的な評価
python -m src.evaluation models/best_model.zip

# 詳細評価（可視化付き）
python -m src.evaluation models/best_model.zip --render --episodes 20

# インタラクティブ評価
python -m src.evaluation models/best_model.zip --interactive
```

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

2. **PyBullet初期化エラー**
   ```
   PhysicsInitializationError: Physics engine initialization failed
   ```
   → Dockerを使用するか、X11設定を確認

3. **メモリ不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   → `batch_size` を減らすか、`n_envs` を調整

4. **学習が収束しない**
   - 報酬関数の重みを調整
   - `learning_rate` を変更
   - エピソード長を確認

### ログレベルの調整

```yaml
logging:
  debug_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

### 設定の検証

```python
from src.utils.config_validator import validate_config

try:
    validate_config(config)
except ConfigValidationError as e:
    print(f"設定エラー: {e}")
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

- **平均報酬**: 100以上
- **前進速度**: 0.1 m/s以上
- **生存率**: 80%以上（10秒間転倒しない）

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📞 サポート

問題が発生した場合は、以下を確認してください：

1. [トラブルシューティング](#🔧-トラブルシューティング)
2. ログファイル (`logs/`)
3. デバッグ情報の確認

---

**Happy Robot Walking! 🤖🚶‍♂️**
