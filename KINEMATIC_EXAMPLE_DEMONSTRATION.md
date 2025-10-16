# 🦿 運動学的足先接地判定の具体例デモンストレーション

## 📋 概要

関節角度から三角関数を使って足先位置を計算し、地面との距離を把握する運動学的計算の具体例を、実際の数値を使って詳しく説明します。

## 🎯 具体例：Bittleの左前足の計算

### 前提条件

- **ロボットのベース位置**: (0.0, 0.0, 0.12) [メートル]
- **ロボットの姿勢**: 水平（ロール=0, ピッチ=0, ヨー=0）
- **左前足の肩関節角度**: 30度（0.524ラジアン）
- **左前足の膝関節角度**: -45度（-0.785ラジアン）

### ステップ1: パラメータの取得

```python
# 運動学的パラメータ（初期化時に設定）
upper_leg_length = 0.49172  # 49.172cm（肩から膝までの距離）

# 左前足の肩関節の基準位置（URDFから取得）
leg_base_pos = [-0.44596, 0.52264, -0.02102]  # [x, y, z]

# 左右対称の符号（左側はそのまま）
symmetry_sign = 1.0
```

### ステップ2: 関節角度の調整

```python
# 取得した関節角度
shoulder_angle = 0.524  # 30度（ラジアン）
knee_angle = -0.785     # -45度（ラジアン）

# 左右対称の考慮（左側はそのまま）
shoulder_angle_adj = shoulder_angle * symmetry_sign
shoulder_angle_adj = 0.524 * 1.0 = 0.524  # 30度のまま
```

### ステップ3: 三角関数による位置計算

```python
# Y-Z平面での計算（脚の伸縮方向）
knee_y_offset = upper_leg_length * cos(shoulder_angle_adj)
knee_y_offset = 0.49172 * cos(0.524)
knee_y_offset = 0.49172 * 0.866 = 0.426  # 42.6cm

knee_z_offset = upper_leg_length * sin(shoulder_angle_adj)
knee_z_offset = 0.49172 * sin(0.524)
knee_z_offset = 0.49172 * 0.5 = 0.246    # 24.6cm
```

**計算の意味:**
- `knee_y_offset = 0.426m`: 肩関節から膝関節までのY方向の距離
- `knee_z_offset = 0.246m`: 肩関節から膝関節までのZ方向の距離

### ステップ4: ローカル座標系での足先位置

```python
# ローカル座標系での足先位置
foot_local_x = leg_base_pos[0]                    # -0.44596m
foot_local_y = leg_base_pos[1] + knee_y_offset   # 0.52264 + 0.426 = 0.949m
foot_local_z = leg_base_pos[2] + knee_z_offset    # -0.02102 + 0.246 = 0.225m

foot_local = [-0.44596, 0.949, 0.225]
```

### ステップ5: ワールド座標系への変換

```python
# ロボットのベース位置と姿勢
base_position = [0.0, 0.0, 0.12]
base_orientation = [0, 0, 0, 1]  # クォータニオン（水平姿勢）

# 回転行列（水平姿勢なので単位行列）
rotation_matrix = [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]

# ローカル位置を回転
rotated_pos = rotation_matrix @ foot_local
rotated_pos = [1, 0, 0] @ [-0.44596, 0.949, 0.225] = [-0.44596, 0.949, 0.225]

# ベース位置を加算
foot_world_pos = rotated_pos + base_position
foot_world_pos = [-0.44596, 0.949, 0.225] + [0.0, 0.0, 0.12]
foot_world_pos = [-0.44596, 0.949, 0.345]
```

### ステップ6: 地面との距離計算

```python
# 足先のワールド座標
foot_world_pos = [-0.44596, 0.949, 0.345]

# 地面は z=0 にあると仮定
ground_distance = foot_world_pos[2]
ground_distance = 0.345  # 34.5cm
```

### ステップ7: 接地判定

```python
# 接地判定の閾値
contact_threshold = 0.05  # 5cm

# 接地判定
if ground_distance <= contact_threshold:
    is_contact = True
else:
    is_contact = False

# 結果: ground_distance = 0.345 > 0.05 → 接地していない
is_contact = False
```

## 🔄 複数の足の例

### 4本の足の状態例

```python
# 各足の関節角度と計算結果
legs_data = {
    'left_front': {
        'shoulder_angle': 30,    # 度
        'knee_angle': -45,       # 度
        'ground_distance': 0.345,  # 34.5cm（接地していない）
        'is_contact': False
    },
    'right_front': {
        'shoulder_angle': -20,   # 度
        'knee_angle': 30,        # 度
        'ground_distance': 0.02,   # 2cm（接地している）
        'is_contact': True
    },
    'left_back': {
        'shoulder_angle': 15,    # 度
        'knee_angle': -30,       # 度
        'ground_distance': 0.08,   # 8cm（接地していない）
        'is_contact': False
    },
    'right_back': {
        'shoulder_angle': -25,   # 度
        'knee_angle': 40,        # 度
        'ground_distance': 0.03,   # 3cm（接地している）
        'is_contact': True
    }
}

# 接地している足の数
foot_contact_count = 2  # right_front, right_back
```

## 🎯 報酬計算の具体例

### 報酬の内訳

```python
# 設定値
foot_contact_reward = 3.0      # 足先接地報酬の重み
proper_gait_reward = 2.0       # 適切な歩容報酬の重み
height_reward_weight = 3.0     # 高さ維持報酬の重み

# 1. 足先接地報酬
if 1 <= foot_contact_count <= 4:
    foot_reward = foot_contact_reward * (foot_contact_count / 4.0)
    foot_reward = 3.0 * (2 / 4.0) = 1.5

# 2. 適切な歩容報酬（2-3本で歩行が理想的）
if 2 <= foot_contact_count <= 3:
    gait_reward = proper_gait_reward
    gait_reward = 2.0

# 3. 足先高さ報酬（接地していない足が適切な高さを維持）
target_foot_height = 0.08  # 8cm
height_reward_sum = 0.0

for leg_name, data in legs_data.items():
    if not data['is_contact']:  # 接地していない足
        distance = data['ground_distance']
        height_error = abs(distance - target_foot_height)
        
        if height_error < 0.03:  # 3cm以内
            height_reward = height_reward_weight * (1.0 - height_error / 0.03)
            height_reward_sum += height_reward

# left_front: height_error = |0.345 - 0.08| = 0.265 > 0.03 → 報酬なし
# left_back: height_error = |0.08 - 0.08| = 0.0 < 0.03 → 報酬あり
# left_back: height_reward = 3.0 * (1.0 - 0.0 / 0.03) = 3.0

height_reward_sum = 3.0

# 合計報酬
total_reward = foot_reward + gait_reward + height_reward_sum
total_reward = 1.5 + 2.0 + 3.0 = 6.5
```

## 📊 デバッグ情報の例

```python
debug_info = {
    'kinematic_foot_contact': {
        'ground_distances': {
            'left_front': 0.345,
            'right_front': 0.02,
            'left_back': 0.08,
            'right_back': 0.03
        },
        'foot_contact_count': 2,
        'foot_details': {
            'left_front': False,
            'right_front': True,
            'left_back': False,
            'right_back': True
        },
        'contact_threshold': 0.05
    }
}
```

## 🔄 学習過程での変化例

### 初期段階（学習開始時）

```python
# ランダムな関節角度
legs_data_initial = {
    'left_front': {'ground_distance': 0.15, 'is_contact': False},
    'right_front': {'ground_distance': 0.12, 'is_contact': False},
    'left_back': {'ground_distance': 0.18, 'is_contact': False},
    'right_back': {'ground_distance': 0.20, 'is_contact': False}
}
# 接地している足: 0本 → 報酬: 0
```

### 中間段階（学習中）

```python
# 一部の足が接地
legs_data_middle = {
    'left_front': {'ground_distance': 0.03, 'is_contact': True},
    'right_front': {'ground_distance': 0.08, 'is_contact': False},
    'left_back': {'ground_distance': 0.02, 'is_contact': True},
    'right_back': {'ground_distance': 0.10, 'is_contact': False}
}
# 接地している足: 2本 → 報酬: 1.5 + 2.0 = 3.5
```

### 最終段階（学習完了）

```python
# 適切な歩行姿勢
legs_data_final = {
    'left_front': {'ground_distance': 0.02, 'is_contact': True},
    'right_front': {'ground_distance': 0.08, 'is_contact': False},  # 持ち上げ
    'left_back': {'ground_distance': 0.08, 'is_contact': False},     # 持ち上げ
    'right_back': {'ground_distance': 0.03, 'is_contact': True}
}
# 接地している足: 2本 → 報酬: 1.5 + 2.0 + 3.0 = 6.5
```

## 🎯 物理エンジン接触検知との比較

### 物理エンジン接触検知の場合

```python
# 接触点の取得
contact_points = p.getContactPoints(robot_id)

# 膝（knee）の接地を検知
for contact in contact_points:
    link_index = contact[3]
    if link_index in foot_links:  # kneeリンク
        foot_contact_count += 1

# 問題: 膝で立っても報酬が与えられる
```

### 運動学的計算の場合

```python
# 関節角度から足先位置を計算
ground_distances = calculate_foot_ground_distances()

# 足先の高さで接地判定
for leg_name, distance in ground_distances.items():
    if distance <= 0.05:  # 5cm以内
        foot_contact_count += 1

# 利点: 足先の位置を正確に計算し、適切な接地を判定
```

## 📈 学習効果の可視化

### TensorBoardでの監視

```python
# 監視可能なメトリクス
tensorboard_metrics = {
    'foot_contact': 1.5,           # 足先接地報酬
    'proper_gait': 2.0,            # 適切な歩容報酬
    'height_maintenance': 3.0,     # 高さ維持報酬
    'ground_distances': {          # 各足の地面からの距離
        'left_front': 0.345,
        'right_front': 0.02,
        'left_back': 0.08,
        'right_back': 0.03
    }
}
```

## 🎉 まとめ

この運動学的計算により：

1. **正確な足先位置**: 関節角度から三角関数で正確に計算
2. **適切な接地判定**: 足先の高さで接地を判定
3. **段階的な報酬**: 接地、歩容、高さの3段階で報酬設計
4. **デバッグの容易さ**: 距離情報を可視化して学習過程を監視

この実装により、ロボットは膝で這うのではなく、適切な姿勢で足先を使って歩行することを学習できます！

---

**実装の核心**: 関節角度 → 三角関数 → 足先位置 → 地面距離 → 接地判定 → 報酬計算

この一連の流れにより、物理シミュレーションに依存せずに、正確な足先接地判定を実現しています。
