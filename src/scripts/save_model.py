import yaml
import argparse
from pathlib import Path
from src.lit_models.utils import get_model, to_onnx

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Load model from YAML configuration")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the ONNX model")
    args = parser.parse_args()

    # YAMLファイルの読み込み
    config = load_yaml_config(args.config_path)
    print(config)
    
    # モデルの取得
    model_params = {k: v for k, v in config.items() if k != "name"}
    model = get_model(config['name'], **model_params)
    
    # 必要に応じてモデルをONNX形式で保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルをONNX形式で保存するための関数
    to_onnx(model, output_dir)

if __name__ == "__main__":
    main()