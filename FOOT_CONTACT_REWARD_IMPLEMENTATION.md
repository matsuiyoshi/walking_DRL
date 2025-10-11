# 🦿 足先接地報酬の実装ガイド

## 📋 概要

四足歩行ロボット（Bittle）が肘・膝で這うような歩行ではなく、**足先で適切に歩行する**ようにするための報酬関数を実装しました。

## 🔍 問題の分析

### 以前の動作
- ロボットが肘と膝（関節の中間部分）で地面を這うように移動
- 足先（エンドエフェクタ）を使わない非効率な歩行
- 人間で例えると、手の肘と足の膝で這っている状態

### 原因
1. **足先接地の報酬がない**: どんな姿勢でも前進すれば報酬
2. **接触情報を使用していない**: PyBulletの接触検知機能を活用していない
3. **関節位置の制約なし**: 適切な姿勢を奨励していない

## 🛠️ 実装内容

### 1. 設定ファイルの更新

**`configs/production.yaml`に追加された報酬パラメータ:**

```yaml
rewards:
  # 既存の報酬
  forward_velocity_weight: 10.0
  survival_reward: 1.0
  fall_penalty: -50.0
  energy_efficiency_weight: -0.01
  orientation_stability_weight: 0.5
  
  # 新規追加: 足先接地報酬
  foot_contact_reward: 3.0           # 足先が接地している報酬
  proper_gait_reward: 2.0            # 適切な歩容報酬（2-3本足接地）
  body_clearance_reward: 5.0         # 胴体・中間関節が地面から離れている報酬
  target_height: 0.12                # 目標胴体高さ（12cm）
  height_reward_weight: 3.0          # 高さ維持報酬の重み
```

### 2. 環境コードの更新（`src/environment.py`）

#### **追加メソッド1: `_identify_foot_links()`**

```python
def _identify_foot_links(self):
    """
    足先リンク（エンドエフェクタ）を特定
    
    Bittleの構造:
    - 各脚に2つの関節（shoulder, knee）
    - kneeリンクが足先（エンドエフェクタ）
    """
```

**機能:**
- URDFから自動的に足先リンク（knee）を検出
- 胴体リンク（base, battery, cover, shoulder）を分類
- 4本の足が正しく検出されたか検証

#### **追加メソッド2: `_calculate_foot_contact_reward()`**

```python
def _calculate_foot_contact_reward(self) -> Tuple[float, Dict]:
    """
    足先接地報酬の計算
    
    Returns:
        Tuple[float, Dict]: (合計報酬, 報酬詳細)
    """
```

**報酬の内訳:**

| 報酬種類 | 条件 | 報酬値 | 効果 |
|---------|------|--------|------|
| **足先接地報酬** | 1-4本の足が接地 | `3.0 * (接地数/4)` | 足先を使うことを奨励 |
| **適切な歩容報酬** | 2-3本の足が接地 | `2.0` | 安定した歩行を奨励 |
| **胴体クリアランス報酬** | 胴体が地面から離れている | `5.0` | 這わないことを奨励 |
| **胴体クリアランスペナルティ** | 胴体が地面に接触 | `-5.0` | 這うことをペナルティ |
| **高さ維持報酬** | 目標高さ±3cm以内 | `3.0 * (1 - 誤差/0.03)` | 適切な高さを維持 |

#### **統合: `_calculate_reward_detailed()`に追加**

既存の報酬計算に足先接地報酬を統合：

```python
# 6. 足先接地報酬（新規追加）
if hasattr(self, 'foot_links'):
    foot_reward, foot_breakdown = self._calculate_foot_contact_reward()
    reward_breakdown.update(foot_breakdown)
```

### 3. 互換性の保証

**既存機能への影響: なし**
- デフォルト値は`0.0`（既存の設定ファイルでは無効）
- `production.yaml`でのみ有効化
- 既存の報酬計算ロジックは一切変更なし

**後方互換性:**
- `hasattr(self, 'foot_links')`チェックで安全に統合
- 足先接地報酬が設定されていない場合は自動的にスキップ

## 🎯 期待される効果

### 学習後の動作
1. **足先での歩行**: 4つのkneeリンク（足先）が地面に接触して歩行
2. **胴体の浮上**: 胴体が地面から12cm程度の高さを維持
3. **中間関節の保護**: shoulderリンク（肘に相当）が地面に接触しない
4. **適切な歩容**: 2-3本の足で交互に歩く動作を学習

### 報酬の働き

```
【従来の動作】
- 前進速度のみで報酬
- どんな姿勢でもOK
→ 肘・膝で這うような歩行

【新しい動作】
- 前進速度報酬: +10.0
- 足先接地報酬: +3.0 (4本接地時)
- 適切な歩容: +2.0 (2-3本時)
- 胴体クリアランス: +5.0 (浮いている時)
- 高さ維持: +3.0 (12cm付近)
→ 足先で立って歩行！
```

## 📊 実装の詳細

### URDFリンク構造（Bittle）

```
base-frame-link (胴体)
├── left-front-shoulder-link (左前肩)
│   └── left-front-knee-link (左前足先) ← 接地対象
├── right-front-shoulder-link (右前肩)
│   └── right-front-knee-link (右前足先) ← 接地対象
├── left-back-shoulder-link (左後肩)
│   └── left-back-knee-link (左後足先) ← 接地対象
└── right-back-shoulder-link (右後肩)
    └── right-back-knee-link (右後足先) ← 接地対象
```

### 接触検知の仕組み

```python
# PyBulletの接触点取得
contact_points = p.getContactPoints(self.robot_id)

for contact in contact_points:
    link_index = contact[3]  # リンクインデックス
    
    if link_index in self.foot_links:
        # 足先接地 → 報酬
        foot_contact_count += 1
    elif link_index in self.body_links:
        # 胴体接触 → ペナルティ
        body_contact = True
```

## 🚀 使い方

### 学習の実行

```bash
# 新しい報酬関数で学習を開始
python -m src.training configs/production.yaml

# 前回のbest_modelは自動的にバックアップされます
# models/best_model_archive/best_model_20251006_190411.zip
```

### 評価

```bash
# 新しいモデルの評価
python -m src.evaluation models/best_model.zip --episodes 20 --render

# 過去のモデルとの比較
python -m src.evaluation models/best_model_archive/best_model_20251006_190411.zip --episodes 20
```

### TensorBoardで確認

```bash
tensorboard --logdir=logs/tensorboard --port=6006

# 以下のメトリクスを確認:
# - foot_contact: 足先接地報酬
# - proper_gait: 適切な歩容報酬
# - body_clearance: 胴体クリアランス報酬
# - height_maintenance: 高さ維持報酬
```

## 🔧 パラメータ調整

報酬の重みを調整することで、歩行スタイルを変更できます：

### 足先接地を強調したい場合
```yaml
foot_contact_reward: 5.0          # 3.0 → 5.0
body_clearance_reward: 8.0        # 5.0 → 8.0
```

### よりダイナミックな歩容にしたい場合
```yaml
proper_gait_reward: 4.0           # 2.0 → 4.0 (2-3本足を強く奨励)
foot_contact_reward: 2.0          # 3.0 → 2.0 (4本接地を緩和)
```

### 高さを厳密に制御したい場合
```yaml
height_reward_weight: 5.0         # 3.0 → 5.0
target_height: 0.15               # 0.12 → 0.15 (より高く)
```

## 📈 学習曲線の見方

### 正常な学習の兆候
- `foot_contact`が徐々に増加
- `body_clearance`がプラスで安定
- `height_maintenance`が0付近で安定
- `forward_velocity`も維持

### 問題の兆候
- `body_clearance`が常にマイナス → 這っている
- `foot_contact`が0に近い → 足先を使っていない
- `height_maintenance`が大きくマイナス → 低すぎる

## ⚠️ 注意事項

### 学習時間
- 新しい報酬関数により、初期段階では報酬が下がる可能性があります
- 以前の這う動作よりも難しい動作を学習するため、1000万ステップ以上の学習を推奨

### ハイパーパラメータ
現在の設定は最適化されていますが、必要に応じて調整してください：
- `learning_rate`: 0.0001 (安定した学習)
- `n_epochs`: 4 (過学習防止)
- `gamma`: 0.995 (長期報酬重視)

### デバッグ
報酬の詳細はログで確認できます：
```python
self.debug_info['last_foot_contact'] = {
    'foot_contact_count': 2,
    'body_contact': False,
    'foot_details': {'foot_4': True, 'foot_6': True}
}
```

## ✅ 実装チェックリスト

- [x] URDFファイルを分析して足先リンクを特定
- [x] 設定ファイルに新しい報酬パラメータを追加
- [x] 足先リンク検出機能を実装
- [x] 足先接地報酬の計算ロジックを実装
- [x] 既存の報酬関数に統合
- [x] 実装のテストと動作確認

## 🎉 まとめ

この実装により、Bittleロボットは：
1. **足先で地面を歩く**ようになります
2. **胴体を浮かせた**適切な姿勢を維持します
3. **効率的な四足歩行**を学習します

学習を実行して、新しい歩行動作を確認してください！

```bash
python -m src.training configs/production.yaml
```

---

**実装者ノート**: この実装は既存のコードとの完全な互換性を保ちながら、段階的に拡張できる設計になっています。将来的には、より高度な歩容パターン（トロット、ギャロップなど）も追加可能です。

