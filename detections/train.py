import argparse
from detections.config import cfg, cfg_from_yaml_file
from detections.models import build_detector
from torch.utils.data import DataLoader
from utils.callbacks import LossHistory
from utils.lr_schedule import get_lr_scheduler

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=r'C:\Users\nick1\Desktop\files\codehub\mid-term\cfgs\dets\fasterrcnn.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch_size for training')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=150, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    model = build_detector(cfg)

    start_epoch = args.start_epoch
    end_epoch = args.start_epoch + args.epochs

    batch_size = args.batch_size
    num_workers = args.num_workers

    save_dir = args.save_dir
    input_shape = args.input_shape

    optimizer_type = args.optimizer_type
    init_lr = args.init_lr
    min_lr = args.min_lr
    lr_decay_type = args.lr_decay_type

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    init_lr_fit = min(max(init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, args.epochs)

    for epoch in range(start_epoch, end_epoch):

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=frcnn_dataset_collate)

            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)

    loss_history.writer.close()


if __name__ == '__main__':
    main()
