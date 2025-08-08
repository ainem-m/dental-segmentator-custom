# Dental Segmentator

歯科医療従事者向けのDICOM画像から自動的に歯の3Dモデル（STLファイル）を生成するPythonアプリケーション

## Overview

本システムは、Zenodoで公開されているdental segmentator（https://zenodo.org/records/10829675）の事前学習済みパラメータを活用し、nnU-NetフレームワークをPyTorchベースで実装しています。これにより、手動でのセグメンテーション作業を大幅に効率化し、診断や治療計画立案、補綴物製作の精度向上を支援します。

## Features

- 🦷 **自動歯科セグメンテーション**: nnU-Netを使用した高精度な歯の自動セグメンテーション
- 🏥 **DICOM対応**: 標準的なDICOM形式の歯科用CT画像に対応
- 🖨️ **3Dプリント対応**: 3Dプリンティングや CAD/CAMシステムで使用可能な高品質STLファイルの生成
- ⚡ **バッチ処理**: 複数のDICOMファイルの自動一括処理
- 🔧 **設定可能**: YAMLファイルによる柔軟な設定管理
- 📊 **処理履歴**: SQLiteデータベースによる処理結果の記録・管理
- 🚀 **マルチGPU対応**: CUDA & MPS (Apple Silicon) による高速処理
- 🧠 **インテリジェント処理**: 自動デバイス選択・メモリ管理・エラー回復
- 🔍 **包括的テスト**: 44%カバレッジの自動テストスイート

## Requirements

- Python 3.9+
- GPU: CUDA対応GPU または Apple Silicon (推奨、CPU処理も可能)
- 8GB以上のメモリ (GPU使用時は追加でGPUメモリが必要)

## Installation

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-org/dental-segmentator-custom.git
cd dental-segmentator-custom
```

### 2. 仮想環境の作成とパッケージインストール (uv使用)

```bash
# uvのインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 3. 事前学習済みモデルのダウンロード

```bash
# モデルの自動ダウンロード (初回実行時)
uv run python -m dental_segmentator.cli download-models
```

## Usage

### 基本的な使用方法

```bash
# 単一DICOMシリーズの処理
uv run python -m dental_segmentator.cli process --input /path/to/dicom/series --output /path/to/output

# フォルダ内の全てのDICOMシリーズを一括処理
uv run python -m dental_segmentator.cli process --input /path/to/dicom/folder --output /path/to/output --batch

# 設定ファイルを指定して処理
uv run python -m dental_segmentator.cli process --config config/custom.yaml --input /path/to/dicom --output /path/to/output
```

### コマンドオプション

```bash
Options:
  --input PATH          入力DICOMファイルまたはディレクトリのパス
  --output PATH         出力STLファイルの保存先ディレクトリ
  --config PATH         設定ファイルのパス (デフォルト: config/default.yaml)
  --batch               バッチ処理モード
  --gpu / --no-gpu      GPU使用の有無 (CUDA/MPS自動検出, デフォルト: 自動選択)
  --log-level LEVEL     ログレベル (DEBUG|INFO|WARNING|ERROR)
  --help               ヘルプメッセージの表示
```

### 設定ファイルのカスタマイズ

`config/default.yaml`をコピーして独自の設定を作成：

```yaml
# config/my-config.yaml
processing:
  input_directory: "./data/input"
  output_directory: "./data/output"
  parallel_jobs: 4

segmentation:
  confidence_threshold: 0.7
  mesh_optimization:
    enable_smoothing: true
    smoothing_iterations: 10

hardware:
  gpu_enabled: true
  gpu_memory_limit: 12288  # MB
```

## Project Structure

```
dental-segmentator-custom/
├── config/                 # 設定ファイル
├── src/                    # ソースコード
│   ├── cli/               # CLIインターフェース
│   ├── processors/        # DICOM処理
│   ├── segmentation/      # nnU-Netセグメンテーション
│   ├── generators/        # STL生成
│   ├── database/          # データベース管理
│   └── utils/            # ユーティリティ
├── models/                # 学習済みモデル (自動ダウンロード)
├── data/                  # データディレクトリ
│   ├── input/            # 入力DICOMファイル
│   ├── output/           # 出力STLファイル
│   └── temp/             # 一時ファイル
├── logs/                  # ログファイル
├── database/              # SQLiteデータベース
└── tests/                 # テストファイル
```

## Configuration

### 主要な設定項目

- **processing.parallel_jobs**: 並列処理数
- **segmentation.confidence_threshold**: セグメンテーションの信頼度閾値
- **hardware.gpu_enabled**: GPU使用の有無
- **logging.level**: ログレベル

詳細な設定項目については `config/default.yaml` を参照してください。

## Output Format

生成されるSTLファイルの形式：

```
output/
├── case_001/
│   ├── tooth_1.stl       # 個別の歯のSTLファイル
│   ├── tooth_2.stl
│   ├── ...
│   └── metadata.json     # 処理結果のメタデータ
└── case_002/
    └── ...
```

## Troubleshooting

### よくある問題

**GPU out of memory エラー**:
- `config.yaml` で `hardware.gpu_memory_limit` を小さくする
- `processing.parallel_jobs` を 1 に設定

**DICOMファイルが認識されない**:
- ファイル拡張子が `.dcm` または `.dicom` になっているか確認
- DICOMファイルのモダリティが歯科用CTであることを確認

**処理が遅い**:
- GPU使用を有効にする (`--gpu` オプション)
- 並列処理数を増やす (`processing.parallel_jobs`)

### ログの確認

```bash
# 詳細なログを出力
uv run python -m dental_segmentator.cli process --log-level DEBUG --input /path/to/dicom --output /path/to/output

# ログファイルの確認
tail -f logs/application.log
```

## Development

### 開発環境のセットアップ

```bash
# 開発用依存関係のインストール
uv sync --dev

# コードフォーマット
uv run black src tests
uv run flake8 src tests

# テスト実行
uv run pytest

# テストカバレッジ
uv run pytest --cov=src --cov-report=html
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

このソフトウェアを研究で使用する場合は、以下を引用してください：

```
@software{dental_segmentator,
  title={Dental Segmentator Custom},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/dental-segmentator-custom}
}
```

また、使用している事前学習済みモデルについても適切に引用してください：
https://zenodo.org/records/10829675

## Acknowledgments

- nnU-Net framework developers
- Zenodo dental segmentator model contributors
- PyTorch community