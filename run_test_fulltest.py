"""
完整测试脚本：
- 解析 `config/test.cfg`（或你指定的 cfg）
- 使用 `utils.niiDataset` 构造 `test_loader`
- 用 `UNet` 构造模型并加载权重（支持 state_dict 或完整模型）
- 调用 `test()`（来自 `test_run.py`）并把预测与指标保存到指定目录

用法示例：
python run_test_fulltest.py --config config/test.cfg --target A --model_path pretrain_model/sam_med2d/sam-med2d_b.pth --save_path ./results

注意：
- 该脚本假设数据目录结构和 `niiDataset` 要求一致（参考 `utils.niiDataset`）
- `test()` 的行为取决于仓库中 `test_run.py` 的实现，本脚本尽量按其期望传入参数
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from utils import parse_config, niiDataset
from unet import UNet
from test_run import test


def build_test_loader(config, dataset, target):
    data_root = config['train'].get('data_root_mms')
    if not data_root:
        raise ValueError('config["train"]["data_root_mms"] 未设置')
    test_img = os.path.join(data_root, 'test', 'img', str(target))
    test_lab = os.path.join(data_root, 'test', 'lab', str(target))
    if not os.path.exists(test_img):
        raise FileNotFoundError(f'test img 路径不存在: {test_img}')
    if not os.path.exists(test_lab):
        raise FileNotFoundError(f'test lab 路径不存在: {test_lab}')
    ds = niiDataset(test_img, test_lab, dataset=dataset, target=target, phase='test')
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    return loader


def load_model_from_path(config, device, model_path=None):
    model = UNet(config).to(device)
    if model_path is None:
        model_path = config['train'].get('source_model_root_mms')
    if model_path is None:
        print('未指定模型路径，返回已初始化模型（随机权重）')
        return model
    if not os.path.exists(model_path):
        print(f'指定的模型文件不存在: {model_path}\n返回已初始化模型（随机权重）')
        return model
    # 尝试加载权重（支持 state_dict 或整 model）
    try:
        loaded = torch.load(model_path, map_location='cpu')
        if isinstance(loaded, dict) and any(k.startswith('_') or k.startswith('enc') or 'state_dict' in k or 'aux' in k for k in loaded.keys()):
            # 很可能是 state_dict
            model.load_state_dict(loaded, strict=False)
            print('已按 state_dict 加载模型权重（strict=False）')
        else:
            # 可能是整个模型对象
            try:
                model = loaded.to(device)
                print('模型文件包含完整模型对象，已直接加载并转移到 device')
            except Exception:
                # 尝试按 state_dict 再加载一次（兼容性）
                model.load_state_dict(loaded, strict=False)
                print('已按 state_dict 加载模型权重（fallback）')
    except Exception as e:
        print('加载模型时出错：', e)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test.cfg', help='cfg path')
    parser.add_argument('--target', type=str, required=True, help='测试目标域，例如 A/B/C/D')
    parser.add_argument('--model_path', type=str, default=None, help='覆盖配置中的模型路径')
    parser.add_argument('--save_path', type=str, default='./results', help='结果保存根目录')
    parser.add_argument('--gpu', type=int, default=None, help='可选：覆盖 cfg 中的 GPU id')
    args = parser.parse_args()

    config = parse_config(args.config)
    dataset = config['train'].get('dataset', 'mms')
    if args.gpu is not None:
        config['train']['gpu'] = args.gpu

    # device
    use_cuda = torch.cuda.is_available() and int(config['train'].get('gpu', 0)) >= 0
    if use_cuda:
        device = torch.device(f"cuda:{config['train']['gpu']}")
    else:
        device = torch.device('cpu')

    # 构造 test_loader
    test_loader = build_test_loader(config, dataset, args.target)
    print('test_loader 构建完成，样本数：', len(test_loader.dataset))

    # 加载模型
    model = load_model_from_path(config, device, args.model_path)
    model.eval()

    # 准备 list_data 与保存目录
    list_data = []
    target = args.target
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # 调用 test()
    list_data = test(config, model, None, test_loader, list_data, target, save_path)
    print('test() 完成，结果保存到：', save_path)

    print(f'\nTarget domain {target}, Model_Path: {args.model_path}\n')
    print(f'Dice:')
    # 返回 list_data: ['90.38±6.53 84.5±4.85 86.12±8.11', '87.0±6.49', '0.62±0.93 0.9±1.29 0.73±0.78', '0.75±1.0']
    print(f'Mean Dice ± Std Dev: {list_data[1]}')
    class_dice = list_data[0].split()
    class_assd = list_data[2].split()
    for i, cls in enumerate(['LV', 'MYO', 'RV']):
        print(f' {cls}: {class_dice[i]}')
    print(f'Assd:')
    for i, cls in enumerate(['LV', 'MYO', 'RV']):
        print(f' {cls}: {class_assd[i]}')

if __name__ == '__main__':
    main()
