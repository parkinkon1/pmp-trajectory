'''
자율주행차량의 미래 점유확률예측 모델 개발
'''
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
from tqdm import tqdm

import torch
import numpy as np
import pytorch_lightning as pl

import math
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler


'''
NOTE: Model training
'''
def train(cfg, model, train_dataloader, val_dataloader):
    print('[+] Model training start!')
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    train_callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=None if cfg.train.dirpath=='' else cfg.train.dirpath,
        filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
        verbose=cfg.train.verbose,
        save_last=cfg.train.save_last,
        save_top_k=cfg.train.save_top_k,
        monitor=cfg.train.monitor,
        mode=cfg.train.mode
    )
    train_callbacks.append(checkpoint_callback)

    if cfg.train.early_stopping.active:
        early_stopping = EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
            mode=cfg.train.early_stopping.mode
        )
        train_callbacks.append(early_stopping)

    trainer_args = {
        'callbacks': train_callbacks,
        'gpus': cfg.train.n_gpus,
        'max_epochs': cfg.train.n_epochs,
        'limit_train_batches': 2000,
        'limit_val_batches': 100,
        'limit_test_batches': 100,
        'profiler': 'advanced'
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


'''
NOTE: Model testing & visualization
'''
def test(cfg, model, test_dataloader):
    print('[+] Start testing!')
    model.cuda()
    model.eval()

    if cfg.test.viz.is_active:
        if cfg.test.viz.custom_destination:
            print('[Manual] Pointing the custom destination points.')
            print('[Manual] (M-Left) Point the destination. / (M-Right) Next batch sample. / (M-Middle) Close testing.')
            global batch, ax
            _, ax = plt.subplots(figsize=(10, 10))
            batch_iter = iter(test_dataloader)
            batch = next(batch_iter)

            import torch.nn.functional as F

            # first image
            with torch.no_grad():
                preds = model({k: v.cuda() for k, v in batch.items() if not k in ['sample_token', 'instance_token']})  # torch.tensor: [B, H, W]
            viz_img = batch['viz'].mean(1).unsqueeze(1).repeat(1,3,1,1) * 0.4  # torch.tensor: [B, C, H, W]
            pred_img = 150 * preds[:, 0, :, :, 1].unsqueeze(1)
            sample_imgs = viz_img.type_as(pred_img) + F.interpolate(torch.cat((pred_img, torch.zeros_like(pred_img).type_as(pred_img).repeat(1,2,1,1)), dim=1), size=viz_img.shape[-2:])

            # sample_imgs = model.get_viz_img(viz_img, preds[:, 0, :, :, 1])  # np.array: [B, H, W, C]
            plt_img = ax.imshow(sample_imgs[0].permute(1,2,0).cpu(), extent=(-25,25,25,-25))
            dest_point = batch['agent_points'][0, 0]
            print('dest:', dest_point)

            def update_predictions(event):
                global batch, ax
                if event.inaxes != ax:
                    return
                # button 1: 마우스 좌클릭
                if event.button == 1:
                    # agent_points = batch['agent_points'].float()  # torch.FloatTensor: [B, S_tgt, 2]
                    batch['agent_points'][0, 0, 0] = event.xdata
                    batch['agent_points'][0, 0, 1] = event.ydata
                    with torch.no_grad():
                        preds = model({k: v.cuda() for k, v in batch.items() if not k in ['sample_token', 'instance_token']})  # torch.tensor: [B, H, W]
                    viz_img = batch['viz'].mean(1).unsqueeze(1).repeat(1,3,1,1) * 0.4  # torch.tensor: [B, C, H, W]
                    pred_img = 150 * preds[:, 0, :, :, 1].unsqueeze(1)
                    sample_imgs = viz_img.type_as(pred_img) + F.interpolate(torch.cat((pred_img, torch.zeros_like(pred_img).type_as(pred_img).repeat(1,2,1,1)), dim=1), size=viz_img.shape[-2:])
                    plt_img.set_data(sample_imgs[0].permute(1,2,0).cpu())
                    dest_point = batch['agent_points'][0, 0]
                    print('dest:', dest_point)
                    plt.draw()
                # button 3: 마우스 우클릭 시 다음 배치로 전환
                if event.button == 3:
                    batch = next(batch_iter)
                    with torch.no_grad():
                        preds = model({k: v.cuda() for k, v in batch.items() if not k in ['sample_token', 'instance_token']})  # torch.tensor: [B, H, W]
                    viz_img = batch['viz'].mean(1).unsqueeze(1).repeat(1,3,1,1) * 0.4  # torch.tensor: [B, C, H, W]
                    pred_img = 150 * preds[:, 0, :, :, 1].unsqueeze(1)
                    sample_imgs = viz_img.type_as(pred_img) + F.interpolate(torch.cat((pred_img, torch.zeros_like(pred_img).type_as(pred_img).repeat(1,2,1,1)), dim=1), size=viz_img.shape[-2:])
                    plt_img.set_data(sample_imgs[0].permute(1,2,0).cpu())
                    dest_point = batch['agent_points'][0, 0]
                    print('dest:', dest_point)
                    plt.draw()
                # 마우스 중간버튼 클릭 시 종료하기
                if event.button == 2:
                    plt.disconnect(cid)
                    plt.close()
            
            cid = plt.connect('button_press_event', update_predictions)
            plt.show()

        else:
            def plot_viz(viz_imgs, axes, n_show):
                for i in range(viz_imgs.shape[0]):
                    axes[i//n_show][i%n_show].imshow(viz_imgs[i])

            batch_size = cfg.dataset.batch_size
            n_show = int(math.sqrt(batch_size))
            _, axes = plt.subplots(n_show, n_show, figsize=(10, 10))
            _ = [[axes[i, j].get_xaxis().set_visible(False), axes[i, j].get_yaxis().set_visible(False)] for j in range(n_show) for i in range(n_show)]
            
            for idx, batch in enumerate(test_dataloader):
                with torch.no_grad():
                    preds = model({k: v.cuda() for k, v in batch.items() if not k in ['sample_token', 'instance_token']})  # torch.tensor: [B, H, W]
                viz_img = batch['viz']  # torch.tensor: [B, C, H, W]
                sample_imgs = model.get_viz_img(viz_img, preds[:, 0, :, :, 1])  # np.array: [B, H, W, C]
                plot_viz(sample_imgs, axes, n_show=n_show)
                plt.pause(1 / cfg.test.viz.viz_fps)
    else:
        trainer = pl.Trainer(resume_from_checkpoint=cfg.test.ckpt_path)
        trainer.test(model, test_dataloader)
    print('[-] Testing ended.')


def split_dataset(dataset, batch_size, validation_split=0.2, shuffle_dataset=True, random_seed=20204333, num_workers=10):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return train_dataloader, val_dataloader

def get_dataloader(cfg):
    if cfg.dataset.type == 'waymo':
        from datautil.waymo_dataset import WaymoDataset
        from datautil.waymo_rast_dataset import waymo_raster_collate_fn
        from torch.utils.data import DataLoader

        pwd = hydra.utils.get_original_cwd() + '/'
        print('Current Path: ', pwd)

        dataset_train = WaymoDataset(pwd+cfg.dataset.waymo.train.tfrecords, pwd+cfg.dataset.waymo.train.idxs)
        dataset_valid = WaymoDataset(pwd+cfg.dataset.waymo.valid.tfrecords, pwd+cfg.dataset.waymo.valid.idxs)
        dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.waymo.batchsize, num_workers=cfg.dataset.waymo.num_workers, collate_fn=lambda x: waymo_raster_collate_fn(x))
        dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.waymo.batchsize, num_workers=cfg.dataset.waymo.num_workers, collate_fn=lambda x: waymo_raster_collate_fn(x))
        
        return dloader_train, dloader_valid

    elif cfg.dataset.type == 'carla':
        from dataset import DatasetCarla
        dataset = DatasetCarla(root_dir=cfg.dataset.carla.root_dir, type='train')
        train_dataloader, val_dataloader = split_dataset(dataset, cfg.dataset.carla.batch_size, validation_split=cfg.dataset.carla.validation_split, 
            shuffle_dataset=cfg.dataset.carla.shuffle_dataset, random_seed=cfg.seed, num_workers=cfg.dataset.carla.num_workers)
        return train_dataloader, val_dataloader
    elif cfg.dataset.type == 'nuscenes':
        from dataset import DatasetNuscenes
        batch_size = cfg.dataset.nuscenes.batch_size
        num_workers = cfg.dataset.nuscenes.num_workers
        train_dataset = DatasetNuscenes(load_dir=cfg.dataset.nuscenes.load_dir, root_dir=cfg.dataset.nuscenes.root_dir, data_type='train')
        val_dataset = DatasetNuscenes(load_dir=cfg.dataset.nuscenes.load_dir, root_dir=cfg.dataset.nuscenes.root_dir, data_type='train')
        dataset_size = len(train_dataset)
        indices = np.arange(dataset_size)
        split = int(np.floor(cfg.dataset.nuscenes.validation_split * dataset_size))
        if cfg.dataset.nuscenes.shuffle_dataset :
            np.random.seed(cfg.seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_dataset.set_indexes(train_indices)
        val_dataset.set_indexes(val_indices)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        return train_dataloader, val_dataloader
    else:
        raise NotImplementedError(f'[!] Dataset {cfg.dataset.type} not found.')
    

def get_model(cfg, load_from_ckpt=False):
    if cfg.model_type == 'SceneTransformer':
        pass
    elif cfg.model_type == 'BasicPMP':
        from models import BasicPMP
        if not load_from_ckpt:
            return BasicPMP(n_channels=6, img_input_dim=(500, 500), img_out_dim=(28, 28), lr=cfg.train.lr)
        else:
            return BasicPMP.load_from_checkpoint(cfg.test.ckpt_path)
    else:
        raise NotImplementedError(f'[!] Model type {cfg.model_type} not found.')


@hydra.main(config_path="conf", config_name="config_pmp")
def main(cfg : DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    train_dataloader, val_dataloader = get_dataloader(cfg)
    if cfg.mode == 'train':
        train(cfg, get_model(cfg, load_from_ckpt=False), train_dataloader, val_dataloader)
    elif cfg.mode == 'test':
        test(cfg, get_model(cfg, load_from_ckpt=True), val_dataloader)
    else:
        raise NotImplementedError(f'[!] Mode {cfg.mode} not found.')


if __name__ == "__main__":
    main()




