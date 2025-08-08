# Requirements Document

## Introduction

このプロジェクトは、歯科医療従事者（歯科医師、歯科技工士、研究者）が、DICOM形式の歯科画像データから自動的に歯の3Dモデル（STLファイル）を生成するためのPythonアプリケーションを開発することを目的とします。

本システムは、Zenodoで公開されているdental segmentator（https://zenodo.org/records/10829675）の事前学習済みパラメータを活用し、nnU-NetフレームワークをPyTorchベースで実装します。これにより、手動でのセグメンテーション作業を大幅に効率化し、診断や治療計画立案、補綴物製作の精度向上を支援します。

このドキュメントでは以下を想定します：

- 入力データは標準的なDICOM形式の歯科用CT画像である
- 出力するSTLファイルは3Dプリンティングや CAD/CAMシステムで使用可能な品質である
- ユーザーはPythonの基本的な知識を持つ歯科医療従事者または技術者である
- 処理は単一マシン上でバッチ処理として実行される

## Requirements

### Requirement 1

**User Story:** As a dental professional, I want to process multiple DICOM files in a specified folder automatically, so that I can generate 3D dental models efficiently without manual intervention for each case.

#### Acceptance Criteria

1. システムは指定されたフォルダ内の全てのDICOMファイルを自動検出できること
1. 各DICOMシリーズに対して個別のSTLファイルを生成できること
1. 処理の進行状況をコンソールまたはログファイルで確認できること
1. 不正なDICOMファイルがあっても他のファイルの処理を継続できること

### Requirement 2

**User Story:** As a system administrator, I want to configure the dental segmentator parameters and model paths, so that I can customize the segmentation behavior for different clinical requirements.

#### Acceptance Criteria

1. 設定ファイル（JSON/YAML形式）でnnU-Netのパラメータを変更できること
1. dental segmentatorの学習済みモデルファイルのパスを設定できること
1. 出力ディレクトリとファイル命名規則を設定できること
1. GPU使用の有無を設定できること（CUDA/MPS/CPU自動選択対応）

### Requirement 3

**User Story:** As a dental technician, I want the generated STL files to be anatomically accurate and suitable for 3D printing, so that I can create precise dental prosthetics and surgical guides.

#### Acceptance Criteria

1. 生成されるSTLファイルは標準的な3Dソフトウェアで開けること
1. メッシュの品質が3Dプリンティングに適していること（水密性、適切なポリゴン数）
1. 実際の歯の解剖学的形状を正確に再現していること
1. ファイルサイズが実用的な範囲内であること（通常10MB以下）

### Requirement 4

**User Story:** As a researcher, I want to validate the segmentation results, so that I can ensure the quality and accuracy of the automated segmentation before using the results for clinical applications.

#### Acceptance Criteria

1. 各処理に対してセグメンテーションの信頼度スコアを出力できること
1. 処理結果の統計情報（検出された歯の数、体積など）を取得できること
1. 元のDICOMファイルと生成されたSTLファイルの対応関係を記録できること
1. エラーが発生した場合の詳細なログを出力できること

### Requirement 5

**User Story:** As an IT support staff, I want the application to have proper error handling and logging, so that I can troubleshoot issues efficiently and maintain system reliability.

#### Acceptance Criteria

1. 全ての例外とエラーが適切にキャッチされ、ログに記録されること
1. ログレベル（DEBUG, INFO, WARNING, ERROR）を設定できること
1. メモリ不足やディスク容量不足などのシステムリソースエラーを検出できること
1. 処理の開始・終了時刻と所要時間を記録できること

### Requirement 6

**User Story:** As a system user, I want the application to have reasonable performance and resource usage, so that I can process dental images efficiently without affecting other system operations.

#### Acceptance Criteria

1. 標準的な歯科CTスキャン（512x512x300程度）を30分以内で処理できること
1. GPU（CUDA/MPS）利用時はCPU版と比較して3倍以上の高速化を実現すること
1. メモリ使用量が利用可能なRAMの80%を超えないこと
1. 複数ファイルの並列処理をサポートすること（リソースに応じて調整可能）

### Requirement 7

**User Story:** As a developer, I want the code to be maintainable and well-documented, so that I can extend the functionality and fix issues efficiently.

#### Acceptance Criteria

1. 主要な関数とクラスにdocstringが記載されていること
1. 設定可能なパラメータと使用方法がREADME.mdに記載されていること
1. 依存関係がrequirements.txtに明記されていること
1. コードがPEP 8スタイルガイドに準拠していること

### Requirement 8

**User Story:** As a quality assurance engineer, I want the application to have automated tests, so that I can verify the functionality works correctly after code changes.

#### Acceptance Criteria

1. 主要な関数に対する単体テストが実装されていること
1. サンプルDICOMデータを使用した結合テストが実装されていること
1. テストカバレッジが80%以上であること
1. CI/CD環境で自動テストが実行可能であること

### Requirement 9

**User Story:** As a security-conscious user, I want the application to handle sensitive medical data securely, so that patient privacy is protected according to healthcare regulations.

#### Acceptance Criteria

1. DICOMファイルの個人識別情報を処理中にログに出力しないこと
1. 一時ファイルは処理完了後に自動削除されること
1. ファイルアクセス権限を適切に設定すること
1. 外部への通信は最小限に抑えること（モデルダウンロード時のみ）

### Requirement 10

**User Story:** As an end user, I want a simple command-line interface, so that I can easily integrate the application into existing workflows and automation scripts.

#### Acceptance Criteria

1. コマンドライン引数で入力フォルダと出力フォルダを指定できること
1. オプションパラメータ（設定ファイルパス、ログレベルなど）を指定できること
1. ヘルプメッセージが表示されること
1. 標準的な終了コード（0: 成功、1: エラー）を返すこと
