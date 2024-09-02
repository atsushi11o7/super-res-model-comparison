import yaml
import argparse
import torch
from pathlib import Path
from src.lit_models.utils import get_model, to_onnx
import torch.quantization
from torch.quantization import get_default_qconfig, prepare, convert

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

    weights_path = "models/espcn_rb/ttaw/checkpoints/epoch=02-val_psnr=28.4373.ckpt"

    if weights_path:
        checkpoint = torch.load(weights_path)
        state_dict = checkpoint['state_dict']
        # state_dictのキーを表示
        print("Keys in state_dict:")
        print(list(state_dict.keys()))

        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(new_state_dict)

        print(f"Model weights loaded from {weights_path}")
    else:
        print("No weights provided, using model with random initialization.")
    
    # 必要に応じてモデルをONNX形式で保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = torch.quantization.quantize_dynamic(
        model,  # 事前トレーニング済みのモデル
        {torch.nn.Conv2d},  # 量子化するレイヤー（例えば、全結合層）
        dtype=torch.qint8  # 量子化後のデータ型
    )
    
    # モデルをONNX形式で保存するための関数
    to_onnx(model, output_dir)

if __name__ == "__main__":
    main()