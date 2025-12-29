import torch
from model.diffusion_model_dgt import DGTDiffusion
from data_provider.conf_gen_cal_metrics import set_rdmol_positions
import copy
from evaluation.eval_functions import get_3D_edm_metric
from data_provider.diffusion_data_module import sample_com_rand_pos
from model.diffusion_model_dgt import remove_mean
from pathlib import Path
from model.diffusion_pl import DiffussionPL, get_precision, disable_compile
import torch.distributed as dist


class UncondGenPL(DiffussionPL):
    def on_validation_epoch_start(self):
        self.test_rdmol_list = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.diffusion_model.train()
        data_batch, selfies_batch = batch
        batch_size = len(data_batch.smiles)
        eval_condition = self.args.mode in {"eval", "eval_test_conform"}

        if dataloader_idx == 0 and not eval_condition:
            train_epoch_condition = (
                self.current_epoch + 1
            ) % self.args.conform_eval_epoch == 0 and self.args.mode == "train"
            with torch.amp.autocast(
                "cuda", dtype=get_precision(self.trainer.precision)
            ):
                loss, lm_loss, diff_loss, _ = self.forward(data_batch, selfies_batch)

            # Use inherited helper methods for consistent val/ prefix
            self.log_val("lm_loss", lm_loss, sync_dist=True, batch_size=batch_size)
            self.log_val("diff_loss", diff_loss, sync_dist=True, batch_size=batch_size)
            self.log_val("loss", loss, sync_dist=True, batch_size=batch_size)

        elif dataloader_idx == 1:

            train_epoch_condition = (
                self.current_epoch + 1
            ) % self.args.test_conform_epoch == 0 and self.args.mode == "train"
            ## inference on the test set, using rdkit predicted conf as input
            if not train_epoch_condition and not eval_condition:
                return
            with torch.amp.autocast(
                "cuda", dtype=get_precision(self.trainer.precision)
            ):
                data_batch, sampled_positions = self.sample(
                    data_batch, selfies_batch, self.infer_time
                )
            sampled_positions = sampled_positions.float().cpu().numpy()
            node_index = 0
            for i in range(len(data_batch.rdmol)):
                smiles = data_batch.smiles[i]
                molecule = copy.deepcopy(data_batch.rdmol[i])
                molecule.RemoveAllConformers()
                num_nodes = molecule.GetNumAtoms()
                positions = sampled_positions[node_index : node_index + num_nodes]
                molecule = set_rdmol_positions(
                    molecule, positions, removeHs=False, add_conformer=True
                )
                self.test_rdmol_list.append((int(data_batch.idx[i]), smiles, molecule))
                node_index += num_nodes

    @torch.no_grad()
    @torch.amp.autocast("cuda", dtype=torch.bfloat16)
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        train_epoch_condition = (
            (self.current_epoch + 1) % self.args.generate_eval_epoch == 0
            and self.current_epoch > 0
            and self.args.mode == "train"
        )
        eval_condition = self.args.mode in {"eval", "eval_test_conform"}
        if train_epoch_condition or eval_condition:
            log_dir = Path(self.logger.log_dir)
            save_path = log_dir / f"predictions_{self.current_epoch}.pt"
            if len(self.test_rdmol_list) > 0:
                if dist.is_initialized():
                    gather_box = [None for _ in range(self.trainer.world_size)]
                    dist.all_gather_object(gather_box, self.test_rdmol_list)
                else:
                    gather_box = [self.test_rdmol_list]
                self.test_rdmol_list = []

                test_rdmol_list = {
                    idx: rdmol for data in gather_box for idx, smiles, rdmol in data
                }
                test_rdmol_list = list(test_rdmol_list.values())

                if self.trainer.is_global_zero:
                    ## save_predictions
                    torch.save(test_rdmol_list, save_path)

                    # if self.args.dataset == "Geom-drugs-jodo":
                    #     # conduct evaluation
                    #     eval_results_3d_unimol, reconstructed_3d_mols = get_3D_edm_metric(test_rdmol_list, self.trainer.datamodule.train_rdmols)
                    #     print(eval_results_3d_unimol)
                    #     eval_results_moses =  self.trainer.datamodule.get_moses_metrics(reconstructed_3d_mols)
                    #     print(eval_results_moses)
                    #     sub_geometry_metric = self.trainer.datamodule.get_sub_geometry_metric(test_rdmol_list)
                    #     print(sub_geometry_metric)

                    #     # Use test/ prefix for generation evaluation metrics
                    #     self.log_test('mol_stable_3d', eval_results_3d_unimol['mol_stable'], sync_dist=True)
                    #     self.log_test('atom_stable_3d', eval_results_3d_unimol['atom_stable'], sync_dist=True)
                    #     self.log_test('validity_3d', eval_results_3d_unimol['Validity'], sync_dist=True)
                    #     self.log_test('novelty_3d', eval_results_3d_unimol['Novelty'], sync_dist=True)
                    #     self.log_test('complete_3d', eval_results_3d_unimol['Complete'], sync_dist=True)

                    #     self.log_test('bond_length_mean', sub_geometry_metric['bond_length_mean'], sync_dist=True)
                    #     self.log_test('bond_angle_mean', sub_geometry_metric['bond_angle_mean'], sync_dist=True)
                    #     self.log_test('dihedral_angle_mean', sub_geometry_metric['dihedral_angle_mean'], sync_dist=True)

                    #     self.log_test('fcd_3d', eval_results_moses['FCD'], sync_dist=True)

    @torch.no_grad()
    def sample(self, data_batch, selfies_batch, T=None):
        num_nodes = data_batch.x.shape[0]
        device = data_batch.x.device
        bs = len(data_batch["smiles"])
        if T is None:
            T = self.noise_scheduler.T
        time_steps = torch.linspace(T, 0.001, self.args.sampling_steps, device=device)
        t_array = time_steps
        s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])

        lm_x = None
        if self.use_llm:
            lm_x, _ = self.forward_llm(data_batch, selfies_batch)

        for i in range(len(t_array)):
            t = t_array[i]
            s = s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t**2 - alpha_t_given_s**2 * sigma_s**2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t
            noise_level = torch.log(alpha_t**2 / sigma_t**2)

            if self.t_cond == "t":
                data_batch["t_cond"] = torch.ones(num_nodes, device=device) * t
            elif self.t_cond == "noise_level":
                data_batch["t_cond"] = (
                    torch.ones(num_nodes, device=device) * noise_level
                )
            else:
                raise NotImplementedError(f"t_cond {self.t_cond} is not implemented")

            data_batch["alpha_t"] = torch.ones((num_nodes, 1), device=device) * alpha_t
            data_batch["sigma_t"] = torch.ones((num_nodes, 1), device=device) * sigma_t
            pred_pos, _ = self.diffusion_model(data_batch, lm_x)

            pos_mean = (
                alpha_t_given_s * sigma_s**2 / sigma_t**2
            ) * data_batch.pos + (alpha_s * sigma2_t_given_s / sigma_t**2) * pred_pos
            pos_mean = remove_mean(pos_mean, data_batch.batch)

            if self.disable_com:
                epsilon_pos = torch.randn(
                    data_batch.pos.shape, device=data_batch.pos.device
                )
            else:
                epsilon_pos = sample_com_rand_pos(
                    data_batch.pos.shape, data_batch.batch
                )
            pos = pos_mean + sigma * epsilon_pos
            assert not torch.isnan(pred_pos).any(), print("here 22", epsilon_pos.shape)
            data_batch["pos"] = pos

        pos = pos_mean * self.pos_std  # std of qm9's dataset
        data_batch["pos"] = pos
        return data_batch, pos

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group()
        parser.add_argument("--llm_model", type=str, default="all_checkpoints/mollama")
        parser.add_argument("--load_llm_ckpt", type=str, default=None)
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--llm_tune", type=str, default="freeze")
        parser.add_argument("--use_llm", action="store_true", default=False)
        parser.add_argument("--llm_cond", action="store_true", default=False)
        parser.add_argument("--tune_embedding", action="store_true", default=False)
        parser.add_argument("--sample_num", type=int, default=10000)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--generate_eval_epoch", type=int, default=10)
        parser.add_argument("--conform_eval_epoch", type=int, default=5)
        parser.add_argument("--test_conform_epoch", type=int, default=20)
        parser.add_argument("--use_llm_projector", action="store_true", default=False)
        parser.add_argument("--use_meta_projector", action="store_true", default=False)
        # 使用自适应metaquery的时候，args.num_metaqueries要设置成：qm9_max_n_nodes=29, geom_drug_max_n_nodes=181
        parser.add_argument("--use_adameta", action="store_true", default=False)
        # 不能太大，MoLlama的max position embeddings只有512
        parser.add_argument("--num_metaqueries", type=int, default=0)
        parser.add_argument("--num_meta_hidden_layers", type=int, default=24)
        # 由于qm9_max_n_nodes=29, geom_drug_max_n_nodes=181，则使用qm9的时候max_n_nodes设置成32，使用geom_drugs的时候max_n_nodes设置成184
        # qm9时num_metaqueries设置成64以上；geom_drugs的时候num_metaqueries设置成256以上
        parser.add_argument(
            "--max_n_nodes",
            type=int,
            default=32,
            help="Maximum number of nodes for the meta2nodes projector output, should be larger than the max nodes in the dataset.",
        )
        parser.add_argument(
            "--meta2nodes_proj_type",
            type=str,
            default="mlp",
            help="Type of projection for meta2nodes, mlp or linear",
        )

        ## llm lora config
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=32)
        parser.add_argument("--lora_dropout", type=int, default=0.1)
        parser.add_argument("--llm_jk", type=str, default="last")
        parser.add_argument("--llm_ckpt", type=str, default=None)

        # optimization
        parser.add_argument(
            "--weight_decay", type=float, default=0.05, help="optimizer weight decay"
        )
        parser.add_argument(
            "--init_lr", type=float, default=1e-4, help="optimizer init learning rate"
        )
        parser.add_argument(
            "--min_lr", type=float, default=1e-5, help="optimizer min learning rate"
        )
        parser.add_argument(
            "--warmup_lr",
            type=float,
            default=1e-6,
            help="optimizer warmup learning rate",
        )
        parser.add_argument(
            "--warmup_steps", type=int, default=1000, help="optimizer warmup steps"
        )
        parser.add_argument(
            "--lr_decay_rate", type=float, default=0.9, help="optimizer lr decay rate"
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            default="linear_warmup_cosine_lr",
            help="type of scheduler",
        )  # or
        parser.add_argument(
            "--not_reduce_node_mean", action="store_true", default=False
        )  # or

        parser.add_argument(
            "--optimizer", type=str, default="adamw", help="type of scheduler"
        )
        parser.add_argument("--init_checkpoint", type=str, default=None)
        parser.add_argument("--ckpt_path", type=str, default=None)

        ## loss
        parser.add_argument("--lm_loss", type=float, default=0)
        parser.add_argument("--diff_loss", type=float, default=1.0)
        parser.add_argument("--save_eval_only", action="store_true", default=False)

        ## diffusion scheduler
        parser.add_argument("--noise_scheduler", type=str, default="cosine")
        parser.add_argument("--continuous_beta_0", type=float, default=0.1)
        parser.add_argument("--continuous_beta_1", type=float, default=20)
        parser.add_argument("--sampling_steps", type=int, default=100)

        ## diffusion parameters
        parser.add_argument("--not_align_loss", action="store_true", default=False)
        parser.add_argument(
            "--translation_correction", action="store_true", default=False
        )
        parser.add_argument("--align_prediction", action="store_true", default=False)
        parser.add_argument("--use_self_att_proj", action="store_true", default=False)
        parser.add_argument("--diff_tune", type=str, default="full")
        parser.add_argument("--diff_ckpt", type=str, default=None)
        DGTDiffusion.add_args(parser)
        return parent_parser
