#!/bin/bash

# Bittle四足歩行ロボット深層強化学習プロジェクト
# Dockerエントリーポイントスクリプト

set -e

echo "=== Bittle Walking DRL Container Starting ==="

# 環境変数の設定
export PYTHONPATH="/app:$PYTHONPATH"
export DISPLAY=${DISPLAY:-:99}

# Xvfbの起動（GUI無しモード用）
if [ "$DISPLAY" = ":99" ]; then
    echo "Starting Xvfb for headless mode..."
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    export DISPLAY=:99
fi

# ログディレクトリの作成
mkdir -p /app/logs
mkdir -p /app/models
mkdir -p /app/evaluation_results

# Pythonパスの確認
echo "Python path: $PYTHONPATH"
echo "Working directory: $(pwd)"

# 引数に応じた処理
case "$1" in
    "train")
        echo "Starting training..."
        python -m src.training --config configs/production.yaml
        ;;
    "debug")
        echo "Starting debug training with rendering..."
        python -m src.training --config configs/debug.yaml
        ;;
    "evaluate")
        echo "Starting evaluation..."
        if [ -z "$2" ]; then
            echo "Error: Model path required for evaluation"
            echo "Usage: docker run ... evaluate <model_path>"
            exit 1
        fi
        # 引数をそのまま渡す
        shift
        python -m src.evaluation "$@"
        ;;
    "test")
        echo "Running tests..."
        cd /app
        python -m pytest tests/ -v
        ;;
    "test-env")
        echo "Running environment tests..."
        cd /app
        python tests/test_environment.py
        ;;
    "interactive")
        echo "Starting interactive evaluation..."
        if [ -z "$2" ]; then
            echo "Error: Model path required for interactive mode"
            echo "Usage: docker run ... interactive <model_path>"
            exit 1
        fi
        # 引数をそのまま渡して、--interactiveを追加
        shift
        python -m src.evaluation "$@" --interactive
        ;;
    "jupyter")
        echo "Starting Jupyter Lab..."
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app
        ;;
    "bash")
        echo "Starting bash shell..."
        exec /bin/bash
        ;;
    "help")
        echo "Available commands:"
        echo "  train           - Start training with default config"
        echo "  evaluate <path> - Evaluate model at <path>"
        echo "  test            - Run all tests"
        echo "  test-env        - Run environment tests only"
        echo "  interactive <path> - Interactive evaluation with GUI"
        echo "  jupyter         - Start Jupyter Lab"
        echo "  bash            - Start bash shell"
        echo "  help            - Show this help"
        ;;
    *)
        if [ $# -eq 0 ]; then
            echo "No command specified. Starting bash shell..."
            echo "Use 'help' to see available commands."
            exec /bin/bash
        else
            echo "Unknown command: $1"
            echo "Use 'help' to see available commands."
            exit 1
        fi
        ;;
esac
