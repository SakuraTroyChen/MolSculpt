import os
import argparse
import torch
import warnings
import lightning as L
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import CSVLogger
from data_provider.data_module import QM9DataModule, QM9LMDataModule, GeomDrugsLMDataModule
from data_provider.geom_drugs_jodo_dm import GeomDrugsJODODM
from model.llm_pl import LLMPL
from datetime import timedelta 

warnings.filterwarnings('ignore', category=UserWarning, message="'has_cuda' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_cudnn' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_mps' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_mkldnn' is deprecated")
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)



def main(args):
    current_epoch = 0
    current_step = 0
    seed = args.seed
    if args.ckpt_path and args.ckpt_path != 'None':
        path = os.path.join(args.ckpt_path, 'checkpoint/mp_rank_00_model_states.pt')
        ckpt = torch.load(path, map_location='cpu')
        current_step = ckpt['global_step']
        seed += current_step
    elif args.init_checkpoint:
        assert (not args.ckpt_path) or args.ckpt_path == 'None'
        ckpt = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
        current_epoch = ckpt['epoch']
        seed += current_epoch
    L.seed_everything(seed)

    # tokenizer
    tokenizer = LLMPL.init_tokenizer(args)
    tokenizer.num_metaqueries = args.num_metaqueries
    tokenizer.use_adameta = args.use_adameta

    print('before inserting total tokens:', len(tokenizer))

    if args.dataset == 'QM9':
        dm = QM9LMDataModule(args.root, args.num_workers, args.batch_size, tokenizer, args)
    elif args.dataset == 'GeomDrugs':
        dm = GeomDrugsLMDataModule(args.root, args.num_workers, args.batch_size, tokenizer, args)
    elif args.dataset == 'GeomDrugs-JODO':
        dm = GeomDrugsJODODM(args.root, args.num_workers, args.batch_size, tokenizer, args)
    else:
        raise NotImplementedError(f"dataset {args.dataset} not implemented")
    print('after inserting total tokens:', len(tokenizer))

    # model
    current_epoch = 0
    if args.init_checkpoint:
        model = LLMPL.load_from_checkpoint(args.init_checkpoint, device=args.devices, strict=True, args=args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens, property_distribution=dm.prop_dist)
        print(f"loading model from {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location='cpu')
        current_epoch = ckpt['epoch']
    else:
        model = LLMPL(args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens, property_distribution=dm.prop_dist)

    print('total params:', sum(p.numel() for p in model.parameters()))

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}',
                                         every_n_epochs=args.save_every_n_epochs,
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/",
                                         filename='last{epoch:02d}',
                                         mode='min',
                                         every_n_epochs=args.cache_epoch,
                                         save_top_k=1,
                                         save_on_train_epoch_end=True,
                                         save_last='link',))

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("multi-gpu training")
        if args.strategy_name == 'deepspeed':
            strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=2, timeout=timedelta(minutes=120))
        else:
            strategy = L.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        print("single-gpu training")
        strategy = 'auto'
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        # limit_val_batches=100,
    )
    if args.mode in ['train']:
        if args.ckpt_path and args.ckpt_path != 'None':
            print('resume training from epoch ', current_epoch)
            trainer.fit_loop.epoch_progress.current.completed = current_epoch
            trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
        else:
            trainer.fit(model, datamodule=dm)
        trainer.validate(model, datamodule=dm)
    elif args.mode in {'eval', 'eval_gen', 'eval_conf', 'eval_1d_gen'}:
        trainer.fit_loop.epoch_progress.current.completed = current_epoch
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='auto')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--delta_train', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--use_flash_attention', action='store_true', default=False)
    parser.add_argument('--cache_epoch', type=int, default=2)
    parser = LLMPL.add_model_specific_args(parser)
    parser.add_argument('--dataset', type=str, default='QM9')
    parser = QM9DataModule.add_model_specific_args(parser)
    parser.add_argument('--condition_property', type=str, default=None)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

