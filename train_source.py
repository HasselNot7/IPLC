from utils import parse_config, set_random,niiDataset
from unet import UNet
from torch.utils.data import DataLoader
import torch
import matplotlib
import os
import argparse
from test_run import test
from metrics import dice_eval
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
matplotlib.use('Agg')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_training_state(model, checkpoint_path, epoch, best_dice, best_model_path):
    state = {
        'epoch': epoch,
        'best_dice': best_dice,
        'best_model_path': best_model_path,
        'model_state': model.state_dict(),
        'enc_opt': model.enc_opt.state_dict(),
        'aux_opt': model.aux_dec1_opt.state_dict(),
        'enc_sched': model.enc_opt_sch.state_dict(),
        'aux_sched': model.dec_1_opt_sch.state_dict(),
    }
    torch.save(state, checkpoint_path)


def load_training_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(model_state, strict=False)
    has_optimizer = 'enc_opt' in checkpoint and 'aux_opt' in checkpoint
    if has_optimizer:
        model.enc_opt.load_state_dict(checkpoint['enc_opt'])
        model.aux_dec1_opt.load_state_dict(checkpoint['aux_opt'])
    if 'enc_sched' in checkpoint:
        model.enc_opt_sch.load_state_dict(checkpoint['enc_sched'])
    if 'aux_sched' in checkpoint:
        model.dec_1_opt_sch.load_state_dict(checkpoint['aux_sched'])
    start_epoch = checkpoint.get('epoch', -1) + 1 if has_optimizer else 0
    best_dice = checkpoint.get('best_dice', 0.)
    best_model_path = checkpoint.get('best_model_path')
    return start_epoch, best_dice, best_model_path, has_optimizer
def get_data_loader(config,dataset,target):
    """
    构建训练、验证和测试数据加载器。
    参数：
        config: 配置字典
        dataset: 数据集名称
        target: 目标域标识
    返回：训练、验证、测试数据加载器
    """
    batch_size = config['train']['batch_size']
    data_root_mms = config['train']['data_root_mms']

    train_img = data_root_mms+'/train/img/{}'.format(target)
    train_lab = data_root_mms+'/train/lab/{}'.format(target)
    valid_img = data_root_mms+'/valid/img/{}'.format(target)
    valid_lab = data_root_mms+'/valid/lab/{}'.format(target)
    test_img = data_root_mms+'/test/img/{}'.format(target)
    test_lab = data_root_mms+'/test/lab/{}'.format(target)

    
    train_test = niiDataset(train_img,train_lab, dataset=dataset, target = target, phase = 'train')
    train_loader = DataLoader(train_test, batch_size = batch_size,shuffle=True, drop_last=True)
    val_dataset = niiDataset(valid_img,valid_lab, dataset=dataset, target = target, phase = 'valid')
    valid_loader = DataLoader(val_dataset, batch_size=1,shuffle=False, drop_last=False)
    test_dataset = niiDataset(test_img,test_lab, dataset=dataset, target = target, phase = 'test')
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False, drop_last=False)
    return train_loader,valid_loader,test_loader

def train(config, train_loader, valid_loader, test_loader, target, list_data, run_id, save_path, resume_checkpoint=None):
    """
    训练主流程，包括模型训练、验证、保存最优模型、测试。
    参数：
        config: 配置字典
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        test_loader: 测试数据加载器
        target: 目标域标识
        list_data: 记录信息列表
    run_id: 当前运行标识（通常为日期字符串）
        save_path: 保存路径
    返回：更新后的list_data
    """
    writer = SummaryWriter(
        log_dir=os.path.join(save_path, "tensorboard", str(target), run_id), comment='')
    directory_path = os.path.join(save_path, 'txt', str(target), run_id)
    ensure_dir(directory_path)
    file_path = os.path.join(directory_path, f'{target}.txt')
    with open(file_path, 'w') as file:
        file.write(run_id + "\n")
    # load exp_name
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    num_classes = config['network']['n_classes_mms']
    # load model
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    iplc_model = UNet(config).to(device)
    iplc_model.train()

    model_dir = os.path.join(save_path, "model", f"{exp_name}_{target}", run_id)
    ensure_dir(model_dir)
    checkpoint_path = os.path.join(model_dir, 'checkpoint-latest.pth')
    auto_resume = config['train'].get('auto_resume', False)
    if resume_checkpoint is None and auto_resume and os.path.exists(checkpoint_path):
        resume_checkpoint = checkpoint_path

    start_epoch = 0
    best_dice = 0.
    best_model_path = None

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"[Resume] Loading checkpoint from {resume_checkpoint}")
        start_epoch, best_dice, best_model_path, has_optimizer = load_training_state(
            iplc_model, resume_checkpoint, device)
        if has_optimizer:
            print(f"[Resume] Continuing from epoch {start_epoch} with best dice {best_dice:.4f}")
        else:
            print('[Resume] Checkpoint has weights only. Starting from epoch 0.')
            start_epoch = 0
    else:
        iplc_model.initialize()
        print("model initialize")
    # load train details
    num_epochs = config['train']['num_epochs']
    valid_epochs = config['train']['valid_epoch']
    j = 0
    for epoch in range(start_epoch, num_epochs):
        iplc_model.train()
        current_loss = 0.
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (B, B_label, _,_) in train_bar:
            B = B.to(device).detach()
            B_label = B_label.to(device).detach()
            loss_seg = iplc_model.train_source(B,B_label)
            current_loss += loss_seg
            train_bar.set_description(f'Epoch [{epoch}/{num_epochs}]')
            train_bar.set_postfix(loss=loss_seg)
        loss_mean = current_loss / (i + 1)
        writer.add_scalar('loss', loss_mean, epoch)
        if (epoch) % valid_epochs == 0:
            current_dice = 0.
            count = -1
            with torch.no_grad():
                iplc_model.eval()
                val_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
                for it,(xt,xt_label,xt_name,lab_Imag) in val_bar:
                    xt = xt.to(device)
                    xt_label = xt_label.numpy().squeeze().astype(np.uint8)
                    output = iplc_model.test_with_name(xt)
                    output = output.squeeze(0)
                    output = torch.argmax(output,dim=1)        
                    output_ = output.cpu().numpy()
                    xt = xt.detach().cpu().numpy().squeeze()
                    output = output_.squeeze()
                    one_case_dice = dice_eval(output,xt_label,num_classes) * 100
                    one_case_dice = np.array(one_case_dice)
                    one_case_dice = np.mean(one_case_dice,axis=0) 
                    current_dice += one_case_dice
                    val_bar.set_description('Validation')
                    val_bar.set_postfix(dice=one_case_dice)
                    count += 1
            dice_mean = current_dice / (count + 1)
            writer.add_scalar('dice', dice_mean, epoch)
            if (current_dice / (count + 1)) > best_dice:
                best_dice = current_dice / (count + 1)
                best_model_path = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), best_dice)
                torch.save(iplc_model.state_dict(), best_model_path)
                torch.save(iplc_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))

        save_training_state(iplc_model, checkpoint_path, epoch, best_dice, best_model_path)
    iplc_model.update_lr()

    final_model_path = None
    if best_model_path and os.path.exists(best_model_path):
        final_model_path = best_model_path
    else:
        latest_path = '{}/model-{}.pth'.format(model_dir, 'latest')
        if os.path.exists(latest_path):
            final_model_path = latest_path
        elif resume_checkpoint and os.path.isfile(resume_checkpoint):
            final_model_path = resume_checkpoint

    if final_model_path and os.path.exists(final_model_path):
        final_state = torch.load(final_model_path, map_location='cpu')
        if isinstance(final_state, dict) and 'model_state' in final_state:
            final_state = final_state['model_state']
        iplc_model.load_state_dict(final_state, strict=False)
    iplc_model.eval()
    # test(config, iplc_model, valid_loader, test_loader, list_data, target, run_id, save_path)
    test(config, iplc_model, valid_loader, test_loader, list_data, target, save_path)
    return list_data


def main():
    """
    主函数，加载配置，循环训练各目标域，保存结果。
    """
    # load config
    save_path = "train_source"
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/train_source.cfg",
                        help='Path to the configuration file')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Checkpoint (.pth) path to resume training from (saves optimizer/lr states).')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Identifier for this run (e.g., 20251212). Defaults to current date when omitted.')
    args = parser.parse_args()
    config = parse_config(args.config)
    resume_path_cfg = config['train'].get('resume_path')
    resume_path = args.resume_path or resume_path_cfg
    run_id = args.run_id or config['train'].get('run_id')
    if resume_path and run_id is None:
        run_id = os.path.basename(os.path.dirname(resume_path.rstrip('/')))
    if not run_id:
        run_id = time.strftime("%Y%m%d", time.localtime())
    list_data = []
    print(config)
    dataset = config['train']['dataset']
    for dataset in ['mms']:
        for target in ['D']:
            config['train']['dataset'] = dataset
            list_data.append(dataset)
            list_data.append(target)
            train_loader,valid_loader,test_loader = get_data_loader(config,dataset,target)
            target_resume_path = None
            if resume_path:
                if '{target}' in resume_path:
                    target_resume_path = resume_path.format(target=target)
                else:
                    target_resume_path = resume_path
            list_data = train(
                config, train_loader, valid_loader, test_loader,
                target, list_data, run_id, save_path, resume_checkpoint=target_resume_path)
            directory_path = os.path.join(save_path, 'txt', str(target), run_id)
            ensure_dir(directory_path)
            file_path = os.path.join(directory_path, f'{target}.txt')
            with open(file_path, 'w') as file:
                for line in list_data:
                    file.write(line + "\n")
        
if __name__ == '__main__':
    # 设置随机种子，保证实验可复现
    set_random()
    main()
