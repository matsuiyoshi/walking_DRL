# 🔄 Best Model バックアップ機能ガイド

## 📋 概要

新しい学習セッションを開始する際、前回の学習で作成された優秀な`best_model.zip`が自動的にバックアップされるようになりました。これにより、報酬設定を調整して再学習する際も、過去の優秀なモデルを失わずに保存できます。

## 🎯 解決する問題

### 以前の動作（問題あり）
```
学習1実行 → best_model.zip 作成（性能: 80点）
学習2実行 → best_model.zip 上書き（性能: 30点）
         → 80点のモデルが失われる！
```

### 新しい動作（改善済み）
```
学習1実行 → best_model.zip 作成（性能: 80点）
学習2開始 → 
  ① 自動バックアップ: best_model_20251006_190411.zip として保存
  ② 新しい学習で best_model.zip を更新（性能: 30点）
  ③ 過去の80点モデルは best_model_archive/ に保護されている！
```

## 🔧 仕組み

### 1. **自動バックアップのタイミング**
- `BittleTrainer`の初期化時（学習開始前）
- 既存の`models/best_model/best_model.zip`が存在する場合のみ実行

### 2. **バックアップ先**
```
models/
├── best_model/
│   └── best_model.zip          # 常に最新の学習の best model
├── best_model_archive/          # 過去のモデルのアーカイブ
│   ├── best_model_20251006_190411.zip  # 学習1の best model
│   ├── best_model_20251007_153020.zip  # 学習2の best model
│   └── best_model_20251008_094530.zip  # 学習3の best model
├── checkpoints/
├── final_model.zip
└── vec_normalize.pkl
```

### 3. **ファイル名の規則**
- 形式: `best_model_YYYYMMDD_HHMMSS.zip`
- タイムスタンプは**ファイルの最終更新日時**から取得
- 同名ファイルが存在する場合は連番を追加（`_1`, `_2`, ...）

## 📖 使い方

### 通常の学習実行
特別な操作は不要です。通常通り学習を実行するだけで自動的にバックアップされます。

```bash
# 学習実行（自動でバックアップされる）
python -m src.training configs/production.yaml
```

### ログの確認
学習開始時のログでバックアップ状況を確認できます：

```
2025-10-09 10:00:23 | bittle_trainer | INFO | === Bittle学習システム初期化開始 ===
2025-10-09 10:00:23 | bittle_trainer | INFO | 前回のbest_modelをバックアップしました
  - original_file: models/best_model/best_model.zip
  - backup_file: models/best_model_archive/best_model_20251006_190411.zip
  - file_size: 168.3 KB
  - timestamp: 20251006_190411
```

初回学習の場合：
```
2025-10-09 10:00:23 | bittle_trainer | INFO | バックアップ対象のbest_modelが存在しません（初回学習）
```

## 🔍 バックアップの確認

### アーカイブ内容の確認
```bash
# バックアップされたモデルのリスト表示
ls -lh models/best_model_archive/

# 出力例:
# -rw-r--r-- 1 root root 168K Oct  6 19:04 best_model_20251006_190411.zip
# -rw-r--r-- 1 root root 172K Oct  7 15:30 best_model_20251007_153020.zip
# -rw-r--r-- 1 root root 169K Oct  8 09:45 best_model_20251008_094530.zip
```

### 過去のモデルを評価
バックアップされたモデルも通常通り評価できます：

```bash
# 2025年10月6日のモデルを評価
python -m src.evaluation models/best_model_archive/best_model_20251006_190411.zip

# 複数のモデルを比較評価
python -m src.evaluation models/best_model_archive/best_model_20251006_190411.zip --episodes 20
python -m src.evaluation models/best_model_archive/best_model_20251007_153020.zip --episodes 20
```

## 🛡️ 安全機能

### 1. **ファイルサイズ検証**
コピー後、元ファイルとバックアップファイルのサイズが一致することを確認します。
不一致の場合は不完全なバックアップを削除し、警告ログを出力します。

### 2. **重複防止**
同じタイムスタンプのファイルが既に存在する場合、自動的に連番を追加します：
```
best_model_20251006_190411.zip
best_model_20251006_190411_1.zip
best_model_20251006_190411_2.zip
```

### 3. **エラーハンドリング**
バックアップに失敗しても学習は継続されます。ただし警告ログで通知されます：
```
WARNING | best_modelのバックアップ中にエラーが発生しました
  - error: [エラー内容]
  - note: 学習は続行されますが、前回のbest_modelが上書きされる可能性があります
```

### 4. **無限ループ防止**
連番生成は最大100回まで試行し、それ以上は中止されます。

## 📊 実装の詳細

### 追加されたメソッド: `_backup_previous_best_model()`

**処理フロー:**
1. `models/best_model/best_model.zip`の存在確認
2. 存在しない場合 → スキップ（初回学習）
3. 存在する場合：
   - バックアップディレクトリ作成: `models/best_model_archive/`
   - ファイルの最終更新日時を取得してタイムスタンプ生成
   - 重複チェック（既存の場合は連番追加）
   - `shutil.copy2()`でコピー（メタデータも保持）
   - ファイルサイズを検証
   - ログ出力

**呼び出しタイミング:**
- `BittleTrainer.__init__()`内
- 設定ファイル読み込み直後
- ディレクトリ作成前

## ✅ 互換性

### 既存機能への影響: なし
- 評価スクリプト: 変更なし（`models/best_model.zip`を参照）
- README例: すべて動作
- EvalCallback: 動作変更なし
- 既存モデル: 自動保護される

### 新規追加
- `import shutil`: 標準ライブラリ（追加インストール不要）
- `_backup_previous_best_model()`メソッド

## 🎯 ベストプラクティス

### 報酬設定を変更して再学習する場合
```bash
# 1. 現在のbest_modelを確認
ls -lh models/best_model/

# 2. 報酬設定を変更
vim configs/production.yaml

# 3. 学習実行（自動でバックアップされる）
python -m src.training configs/production.yaml

# 4. 学習後、過去のモデルと比較
python -m src.evaluation models/best_model/best_model.zip --episodes 20
python -m src.evaluation models/best_model_archive/best_model_20251006_190411.zip --episodes 20
```

### 定期的なクリーンアップ
古いバックアップが不要になったら手動で削除できます：
```bash
# 古いバックアップを削除（例: 2025年10月6日より前）
rm models/best_model_archive/best_model_202510[0-5]*.zip
```

## 📝 まとめ

- ✅ **自動バックアップ**: 学習開始時に自動実行
- ✅ **安全性**: ファイルサイズ検証、エラーハンドリング
- ✅ **互換性**: 既存コードへの影響なし
- ✅ **透明性**: 詳細なログ出力
- ✅ **柔軟性**: 過去のモデルも評価可能

これにより、安心して報酬設定を調整しながら、最適なモデルを探索できます！

