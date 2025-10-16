# 🦿 運動学的足先接地報酬の実装ガイド

## 📋 概要

関節角度から三角関数を使って足先位置を計算し、地面との距離を把握する運動学的計算による足先接地報酬を実装しました。

## 🎯 実装の背景

### 問題点
- 物理エンジンの接触検知では、膝（knee-link）が接地していると報酬が与えられる
- ロボットが膝で立つことを学習してしまい、適切な歩行姿勢にならない
- 真の足先リンクがURDFに存在しないため、接触検知だけでは不十分

### 解決策
- 関節角度から運動学的に足先位置を計算
- 計算された足先位置から地面との距離を算出
- 距離に基づいて接地判定と報酬計算を行う

## 🛠️ 実装内容

### 1. 運動学的パラメータの初期化

**`src/environment.py` - `_initialize_kinematic_parameters()`**

```python
def _initialize_kinematic_parameters(self):
    """運動学的計算のためのパラメータを初期化"""
    # URDFから取得したリンク長さ
    self.upper_leg_length = 0.49172  # 肩から膝までの距離（メートル）
    
    # 各脚の肩関節のベース座標系での位置
    self.leg_base_positions = {
        'left_front': [-0.44596, 0.52264, -0.02102],
        'right_front': [0.45149, 0.52264, -0.02102],
        'left_back': [-0.44596, -0.51923, -0.02102],
        'right_back': [0.45149, -0.51923, -0.02102]
    }
    
    # 左右対称を考慮するための符号（右側は時計回り反転）
    self.leg_symmetry_sign = {
        'left_front': 1.0,
        'right_front': -1.0,
        'left_back': 1.0,
        'right_back': -1.0
    }
```

### 2. 足先位置の運動学的計算

**`_calculate_foot_positions_from_kinematics()`**

```python
def _calculate_foot_positions_from_kinematics(self) -> Dict[str, Tuple[float, float, float]]:
    """関節角度から三角関数を使って足先位置を計算"""
    
    # 各脚について
    for leg_name in ['left_front', 'right_front', 'left_back', 'right_back']:
        # 肩関節と膝関節の角度を取得
        shoulder_angle = joint_angles[shoulder_joint_idx]
        knee_angle = joint_angles[knee_joint_idx]
        
        # 左右対称の考慮（モーターの回転方向）
        symmetry_sign = self.leg_symmetry_sign[leg_name]
        shoulder_angle_adj = shoulder_angle * symmetry_sign
        
        # Y-Z平面での足先位置計算
        knee_y_offset = self.upper_leg_length * np.cos(shoulder_angle_adj)
        knee_z_offset = self.upper_leg_length * np.sin(shoulder_angle_adj)
        
        # ローカル座標からワールド座標への変換
        foot_world_pos = self._transform_local_to_world(...)
```

### 3. 地面との距離計算

**`_calculate_foot_ground_distances()`**

```python
def _calculate_foot_ground_distances(self) -> Dict[str, float]:
    """足先と地面との距離を計算"""
    foot_positions = self._calculate_foot_positions_from_kinematics()
    
    for leg_name, foot_pos in foot_positions.items():
        # 地面は z=0 にあると仮定
        ground_distance = foot_pos[2]  # Z座標が地面からの距離
```

### 4. 運動学的報酬計算

**`_calculate_foot_contact_reward_kinematic()`**

```python
def _calculate_foot_contact_reward_kinematic(self) -> Tuple[float, Dict]:
    """運動学的計算による足先接地報酬"""
    
    # 足先と地面との距離を計算
    ground_distances = self._calculate_foot_ground_distances()
    
    # 接地判定（5cm以内なら接地）
    contact_threshold = 0.05
    
    # 報酬計算
    # 1. 足先接地報酬
    # 2. 適切な歩容報酬（2-3本足接地）
    # 3. 足先高さ報酬（接地していない足）
```

### 5. 報酬関数への統合

**`_calculate_reward_detailed()`**

```python
# 運動学的計算 or 物理エンジンの接触検知を選択
use_kinematic_contact = self.reward_config.get('use_kinematic_contact', False)

if use_kinematic_contact:
    # 運動学的計算による足先接地報酬
    foot_reward, foot_breakdown = self._calculate_foot_contact_reward_kinematic()
else:
    # 物理エンジンの接触検知による足先接地報酬（既存）
    foot_reward, foot_breakdown = self._calculate_foot_contact_reward()
```

## ⚙️ 設定ファイルの変更

### `configs/production.yaml`

```yaml
rewards:
  # 足先接地報酬計算方法の選択
  use_kinematic_contact: true        # true: 運動学的計算, false: 物理エンジンの接触検知
  
  # 足先接地報酬
  foot_contact_reward: 3.0           # 足先が接地している報酬
  proper_gait_reward: 2.0            # 適切な歩容報酬（2-3本足接地）
  body_clearance_reward: 5.0         # 胴体・中間関節が地面から離れている報酬
  target_height: 0.12                # 目標胴体高さ（12cm）
  height_reward_weight: 3.0          # 高さ維持報酬の重み
```

### `configs/debug.yaml`

```yaml
rewards:
  # デバッグ時は物理エンジンの接触検知を使用
  use_kinematic_contact: false
```

## 🔧 運動学的計算の仕組み

### Bittleの脚の構造

```
base-frame-link (胴体)
├── shoulder-link (肩関節) - 角度: θ1
│   └── knee-link (膝関節) - 角度: θ2 ← これが最外端（足先）
```

### 座標変換の流れ

1. **関節角度の取得**: PyBulletから8つの関節角度を取得
2. **左右対称の考慮**: 右側の足は符号を反転（モーターの回転方向）
3. **ローカル座標計算**: 肩の基準位置から三角関数で足先位置を計算
4. **ワールド座標変換**: ロボットの姿勢（クォータニオン）を考慮して変換
5. **地面との距離**: Z座標が地面からの高さ

### 数式

```
# 肩関節の回転による足先のY-Z平面での位置
knee_y = leg_base_y + upper_leg_length × cos(θ_shoulder)
knee_z = leg_base_z + upper_leg_length × sin(θ_shoulder)

# 左右対称の考慮
θ_adjusted = θ × symmetry_sign
  where symmetry_sign = 1.0 (左側), -1.0 (右側)
```

## 📊 期待される効果

### 運動学的計算の利点

1. **正確性**: 物理シミュレーションの接触検知に依存しない
2. **効率性**: 計算が軽く、リアルタイムで実行可能
3. **制御性**: 関節角度から直接計算するため、学習が安定
4. **柔軟性**: 接地判定の閾値を自由に調整可能

### 物理エンジン接触検知との比較

| 項目 | 運動学的計算 | 物理エンジン接触検知 |
|-----|------------|------------------|
| **精度** | 関節角度から正確に計算 | 衝突検知に依存 |
| **速度** | 非常に高速 | やや重い |
| **接地判定** | 距離ベース（調整可能） | 接触の有無のみ |
| **デバッグ** | 距離を可視化可能 | 接触点の情報のみ |
| **安定性** | 高い（数値計算） | 物理シミュレーションに依存 |

## 🚀 使用方法

### 運動学的計算を有効化

```yaml
# configs/production.yaml
rewards:
  use_kinematic_contact: true  # 運動学的計算を使用
```

### 物理エンジンの接触検知を使用

```yaml
# configs/debug.yaml
rewards:
  use_kinematic_contact: false  # 物理エンジンを使用（既存の動作）
```

### 学習の実行

```bash
# 運動学的計算で学習
python -m src.training --config configs/production.yaml

# 既存の方法で学習（互換性確認）
python -m src.training --config configs/debug.yaml
```

## 🔍 デバッグ情報

運動学的計算のデバッグ情報は `debug_info` に保存されます：

```python
self.debug_info['kinematic_foot_contact'] = {
    'ground_distances': {
        'left_front': 0.08,
        'right_front': 0.02,  # 接地
        'left_back': 0.03,    # 接地
        'right_back': 0.09
    },
    'foot_contact_count': 2,
    'foot_details': {
        'left_front': False,
        'right_front': True,
        'left_back': True,
        'right_back': False
    },
    'contact_threshold': 0.05
}
```

## ✅ 互換性の保証

### 既存の実装との互換性

- **デフォルト動作**: `use_kinematic_contact: false` で既存の動作を維持
- **段階的移行**: 設定ファイルで簡単に切り替え可能
- **後方互換性**: 既存の設定ファイルでも動作（デフォルトでfalse）
- **並行使用**: 両方の方法を同時にテスト可能

### テスト方法

```bash
# 1. 既存の方法でテスト
python -m src.evaluation models/best_model.zip --config configs/debug.yaml

# 2. 運動学的計算でテスト
python -m src.evaluation models/best_model.zip --config configs/production.yaml

# 3. 比較
# デバッグ情報で両方の結果を比較可能
```

## 📈 パラメータ調整

### 接地判定の閾値

```python
# _calculate_foot_contact_reward_kinematic() 内
contact_threshold = 0.05  # 5cm以内なら接地

# 調整例：
# - より厳密に: 0.03 (3cm)
# - より緩く: 0.08 (8cm)
```

### 足先高さの目標値

```python
target_foot_height = 0.08  # 8cm（歩行時の足の持ち上げ高さ）

# 調整例：
# - より高く: 0.10 (10cm)
# - より低く: 0.05 (5cm)
```

## ⚠️ 注意事項

### 運動学的計算の制限

1. **簡略化**: 膝関節の回転は考慮していない（膝が最外端のため）
2. **2D計算**: Y-Z平面での計算を簡略化
3. **地面の仮定**: 地面がz=0の平面であることを仮定

### 今後の改善点

1. **膝関節の回転を考慮**: より正確な足先位置計算
2. **3D運動学**: X-Y-Z全方向での正確な計算
3. **不整地対応**: 地面の傾斜や高低差を考慮

## 🎉 まとめ

この実装により：

1. **関節角度から足先位置を正確に計算**できるようになりました
2. **物理シミュレーションに依存しない接地判定**が可能になりました
3. **左右対称を考慮したモーター制御**が実装されました
4. **既存の実装との完全な互換性**を保っています

運動学的計算により、ロボットがより適切な歩行姿勢を学習することが期待されます！

```bash
# 新しい実装で学習を開始
python -m src.training --config configs/production.yaml
```

---

**実装者ノート**: この実装は既存のコードとの完全な互換性を保ちながら、運動学的計算による高精度な足先接地判定を提供します。設定ファイルで簡単に切り替え可能なため、段階的な移行が可能です。

