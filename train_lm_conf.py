import argparse
import torch
import pickle
import warnings
import lightning as L
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import CSVLogger
from model.diffusion_pl import DiffussionPL
from data_provider.diffusion_data_module import QM9TorDFDataModule, GeomDrugsTorDFDataModule
from data_provider.diffusion_data_module_v2 import QM9DataModule
import os
from rdkit import RDLogger
from model.ema import EMACallBack

disable_compile = torch.cuda.get_device_name(0).find('AMD') >= 0

# Suppress PyTorch deprecation warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, message="'has_cuda' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_cudnn' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_mps' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="'has_mkldnn' is deprecated")

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['DEEPSPEED_TIMEOUT'] = '180' # set NCCL timeout to 3 hours instead of 30 minutes to avoid broken pipe error
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
RDLogger.DisableLog('rdApp.*')
# warnings.filterwarnings(action='ignore')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)



def main(args):
    current_epoch = 0
    current_step = 0
    seed = args.seed
    # if args.ckpt_path and args.ckpt_path != 'None':
    #     path = os.path.join(args.ckpt_path, 'checkpoint/mp_rank_00_model_states.pt')
    #     ckpt = torch.load(path, map_location='cpu')
    #     current_step = ckpt['global_step']
    #     seed += current_step
    # elif args.init_checkpoint:
    #     assert (not args.ckpt_path) or args.ckpt_path == 'None'
    #     ckpt = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
    #     current_epoch = ckpt['epoch']
    #     seed += current_epoch
    L.seed_everything(seed)

    # tokenizer
    tokenizer = DiffussionPL.init_tokenizer(args)
    tokenizer.num_metaqueries = args.num_metaqueries
    tokenizer.use_adameta = args.use_adameta
    print('before inserting total tokens:', len(tokenizer))

    if args.dataset == 'QM9-df':
        use_distributed_sampler = True
        if args.condition_property is None:
            dm = QM9TorDFDataModule(args.root, args.num_workers, args.batch_size, tokenizer, args.load_test_only, args)
        else:
            dm = QM9DataModule(args.root, args.num_workers, args.batch_size, tokenizer, args)
    elif args.dataset == 'Geom-drugs-df':
        use_distributed_sampler = False
        dm = GeomDrugsTorDFDataModule(args.root, args.num_workers, args.batch_size, tokenizer, args.load_test_only, args)
    else:
        raise NotImplementedError(f"dataset {args.dataset} not implemented")
    print('after inserting total tokens:', len(tokenizer))

    # model
    if args.init_checkpoint:
        model_state_dict = {}
        checkpoint_path = os.path.join(args.init_checkpoint, 'checkpoint/mp_rank_00_model_states.pt')
        main_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_state_dict.update(main_model['module'])
    
        # if args.use_ema and 'ema_state_dict' in ckpt:
        #     state_dict = ckpt['ema_state_dict']
        # else:
        #     state_dict = ckpt['state_dict']

        model = DiffussionPL(args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens, noise_scheduler=dm.noise_scheduler, pos_std=dm.pos_std, property_normalizations=dm.prop_norms, property_distribution=dm.prop_dist)
        # Load with strict=False to handle missing conditional MLP keys in older checkpoints
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        print(f"Loading {'EMA ' if args.use_ema else ''}model from {args.init_checkpoint}")
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    else:
        model = DiffussionPL(args, tokenizer=tokenizer, max_sf_tokens=dm.max_sf_tokens, noise_scheduler=dm.noise_scheduler, pos_std=dm.pos_std, property_normalizations=dm.prop_norms, property_distribution=dm.prop_dist)
    model.train()

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

    if args.use_ema:
        callbacks.append(EMACallBack(decay=0.9999))

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("multi-gpu training")
        if args.strategy_name == 'deepspeed':
            strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=2)
        else:
            strategy = L.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        print("single-gpu training")
        strategy = 'auto'
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    # Stage 2 Training for 10 epochs: LoRA Weights of MolLLaMA and the cross-modal projector
    if args.delta_train and args.mode == 'train' and args.init_checkpoint is None and (args.ckpt_path is None or args.ckpt_path == 'None'):
        trainer = L.Trainer(
            accelerator=args.accelerator,
            num_nodes=args.num_nodes,
            devices=args.devices,
            precision=args.precision,
            max_epochs=args.delta_max_epochs,
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=5,
            strategy=strategy,
            logger=logger,
            detect_anomaly=args.detect_anomaly,
            enable_checkpointing=False,
            use_distributed_sampler=use_distributed_sampler,
        )
        if args.tune_embedding:
            model.set_trainble_params(['projector', 'global_tokens', 'self_att_proj', 'linear_proj', \
                                    'llm_projector', 'embed_tokens', 'lora_B', 'lora_A'], True)
        else:
            model.set_trainble_params(['projector', 'global_tokens', 'self_att_proj', 'linear_proj', \
                                    'llm_projector', 'lora_B', 'lora_A'], True)
        print(f"=========Delta Train: 10 epochs==========")
        trainer.fit(model, datamodule=dm)
        model.restore_trainble_params(False)
        args.mode = 'train'
        trainer.save_checkpoint(f'./all_checkpoints/{args.filename}/delta.ckpt', weights_only=True)

        del dm # release memory and reinitialize datamodule
        if args.dataset == 'QM9-df':
            if args.condition_property is None:
                dm = QM9TorDFDataModule(args.root, args.num_workers, args.batch_size, tokenizer, False, args)
            else:
                dm = QM9DataModule(args.root, args.num_workers, args.batch_size, tokenizer, args)
        elif args.dataset == 'Geom-drugs-df':
            dm = GeomDrugsTorDFDataModule(args.root, args.num_workers, args.batch_size, tokenizer, False, args)
        else:
            raise NotImplementedError(f"dataset {args.dataset} not implemented")

    ## this is to recompile the code when switching different training methods
    torch._dynamo.reset()

    # if args.mode.find('eval') >= 0:
    #     model.diffusion_model = torch.compile(model.diffusion_model, dynamic=True, fullgraph=False, disable=disable_compile)

    if device_count > 1:
        print("multi-gpu training")
        if args.strategy_name == 'deepspeed':
            strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=2)
        else:
            strategy = L.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        print("single-gpu training")
        strategy = 'auto'

    print(f"============Train: {args.max_epochs} epochs============")
    trainer = L.Trainer(
        num_nodes=args.num_nodes,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        detect_anomaly=args.detect_anomaly,
        use_distributed_sampler=use_distributed_sampler,
        # num_sanity_val_steps=0,
        # limit_train_batches=10,
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
    elif args.mode in {'eval', 'eval_gen', 'eval_conf', 'eval_1d_gen', 'eval_test_conform'}:
        trainer.fit_loop.epoch_progress.current.completed = current_epoch
        trainer.validate(model, datamodule=dm, ckpt_path=args.ckpt_path)

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
    parser.add_argument('--delta_max_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--delta_train', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser = DiffussionPL.add_model_specific_args(parser)
    parser.add_argument('--dataset', type=str, default='QM9-df')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--cache_epoch', type=int, default=5)
    parser.add_argument('--load_test_only', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--use_flash_attention', action='store_true', default=False)
    parser = QM9TorDFDataModule.add_model_specific_args(parser)
    parser.add_argument('--condition_property', type=str, default=None)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

