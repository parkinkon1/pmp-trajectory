import os
import sys
import hydra

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
from model.pl_module import SceneTransformer



def train(cfg, model, train_dataloader, val_dataloader):
    print('[+] Model training start!')
    # print(f'Dataset Size - train: {len(train_dataloader)}, valid: {len(val_dataloader)}')
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    train_callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        filename='{epoch}-{val_loss:.2f}',
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    train_callbacks.append(checkpoint_callback)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    train_callbacks.append(early_stopping)

    trainer_args = {
        'callbacks': train_callbacks,
        'gpus': cfg.train.n_gpus,
        'max_epochs': cfg.train.max_epochs,
        'profiler': 'advanced',
        'log_every_n_steps': 500,
        'gradient_clip_val': 1.0,
        # 'val_check_interval': 100
    }
    if cfg.train.ckpt_path != '':
        trainer_args['resume_from_checkpoint'] = cfg.train.ckpt_path
    if cfg.train.accelerator != '':
        trainer_args['accelerator'] = cfg.train.accelerator

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_dataloader, val_dataloader)
    print('[+] Model training end, start testing...')
    trainer.test(model, val_dataloader)
    print('[-] Traing and testing ended.')



@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids

    pl.seed_everything(cfg.seed)
    pwd = hydra.utils.get_original_cwd() + '/'
    print('Current Path: ', pwd)

    if cfg.mode == 'train':
        dataset_train = WaymoDataset(pwd+cfg.dataset.train.tfrecords, pwd+cfg.dataset.train.idxs)
        dataset_valid = WaymoDataset(pwd+cfg.dataset.valid.tfrecords, pwd+cfg.dataset.valid.idxs)
        dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.batchsize, num_workers=cfg.dataset.num_workers, collate_fn=waymo_collate_fn)
        dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.batchsize, num_workers=cfg.dataset.num_workers, collate_fn=waymo_collate_fn)

        model = SceneTransformer(None, cfg.model.in_feature_dim, cfg.model.in_dynamic_rg_dim, cfg.model.in_static_rg_dim,
                                    cfg.model.time_steps, cfg.model.feature_dim, cfg.model.head_num, cfg.model.k, cfg.model.F)
        train(cfg, model, dloader_train, dloader_valid)
    else:
        raise NotImplementedError(f'Yet Implemented mode: {cfg.mode}')

    return 0

if __name__ == '__main__':
    sys.exit(main())
