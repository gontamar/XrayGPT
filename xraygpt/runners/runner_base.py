"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Import standard modules for handling time, path, logging, and JSON
import datetime  # For handling date and time calculations
import json      # For saving and loading configuration/statistics in JSON format
import logging   # For logging messages to console or files
import os        # For interacting with the operating system (paths, env, etc)
import time      # For timing durations
from pathlib import Path  # For easy path manipulations

# Import torch and distributed computing modules
import torch
import torch.distributed as dist
import webdataset as wds  # For working with WebDataset data pipelines

# Import utility functions for distributed training and registry system
from xraygpt.common.dist_utils import (
    download_cached_file,    # Download and cache files
    get_rank,                # Get the current process rank in distributed setup
    get_world_size,          # Get the number of processes
    is_main_process,         # Check if in main process
    main_process,            # Decorator to run a function only in main process
)
from xraygpt.common.registry import registry  # Global registry for classes/objects
from xraygpt.common.utils import is_url       # Utility to check if string is a URL
from xraygpt.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset  # Dataset utils
from xraygpt.datasets.datasets.dataloader_utils import (
    IterLoader,         # Custom iterator loader
    MultiIterLoader,    # Loader for multiple datasets with sampling ratios
    PrefetchLoader,     # Loader with prefetching for efficiency
)
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP wrapper for models
from torch.utils.data import DataLoader, DistributedSampler   # Data loaders and samplers

# Register this runner class in the registry under the name "runner_base"
@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        # Save configuration, job id, task, datasets, and model
        self.config = cfg
        self.job_id = job_id
        self.task = task
        self.datasets = datasets
        self._model = model

        # Initialize internal variables to None; will be created on demand
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0  # The epoch to start from (for checkpoint resume)

        # self.setup_seeds()  # (Optional) set random seeds for reproducibility
        self.setup_output_dir()  # Prepare output and result directories

    @property
    def device(self):
        # Get the device (cpu/cuda) to use for training
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)
        return self._device

    @property
    def use_distributed(self):
        # Return True if distributed training is enabled
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # Move the model to the appropriate device if it is not already there
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # If distributed training is enabled, wrap with DDP
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu]
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # Create the optimizer if it has not been created yet
        if self._optimizer is None:
            num_parameters = 0  # Count total trainable parameters
            p_wd, p_non_wd = [], []  # Separate param groups for weight decay
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # Skip frozen weights
                print(n)  # Print parameter name (for debugging)
                # No weight decay for biases, LayerNorm, BatchNorm, etc.
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        # If AMP (mixed precision) is enabled, create a GradScaler for it
        amp = self.config.run_cfg.get("amp", False)
        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)
            # Get scheduler hyperparameters from config or properties
            max_epoch = self.max_epoch
            min_lr = self.min_lr
            init_lr = self.init_lr
            # Optional scheduler parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)
            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders['train'])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000  # Fallback value
            # Create the scheduler instance
            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # Log how datasets will be combined
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            # Rearrange datasets by split (e.g., train/val/test)
            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = datasets

            # Print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # Mixed webdataset.DataPipeline and torch Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )
                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # A single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # A single webdataset.DataPipeline (no __len__)
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )
                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # Create dataloaders for each split
            split_names = sorted(self.datasets.keys())
            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]
            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]
            # Gather collate functions for each dataset
            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            # Actually create loaders
            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        # Return True if the model is running on CUDA (GPU)
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        # Return the max number of epochs as integer
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        # Get the logging frequency (how often to log training stats)
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        # Initial learning rate
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        # Minimum learning rate
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        # Return gradient accumulation steps
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        # Return the validation splits
        valid_splits = self.config.run_cfg.get("valid_splits", [])
        if len(valid_splits) == 0:
            logging.info("No validation splits found.")
        return valid_splits

    @property
    def test_splits(self):
        # Return the test splits
        test_splits = self.config.run_cfg.get("test_splits", [])
        return test_splits

    @property
    def train_splits(self):
        # Return the train splits
        train_splits = self.config.run_cfg.get("train_splits", [])
        if len(train_splits) == 0:
            logging.info("Empty train splits.")
        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        # Use distributed sampler for evaluation
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        # Get the checkpoint path to resume from
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        # Return the train dataloader
        train_dataloader = self.dataloaders["train"]
        return train_dataloader

    def setup_output_dir(self):
        # Set up output/result directories for logs and checkpoints
        lib_root = Path(registry.get_path("library_root"))
        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))
        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        # Main training loop over all epochs
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()  # Log the config

        # Resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # Training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # Evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))
                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."
                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics
                                self._save_checkpoint(cur_epoch, is_best=True)
                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)
            else:
                # If no validation, save checkpoint after each epoch
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            if self.config.run_cfg.distributed:
                dist.barrier()  # Sync all processes

        # After training finishes, run evaluation on test set
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def test(self):
        # Run testing only (no training)
        start_time = time.time()
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)
        test_stats = self.test_epoch(1)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Testing time {}".format(total_time_str))

    def evaluate(self, ckpt, cur_epoch="best", skip_reload=False):
        # Evaluate model on all test splits
        test_logs = dict()
        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    ckpt, split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )
            return test_logs

    def train_epoch(self, epoch):
        # Train model for one epoch
        self.model.train()
        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )
    
    def test_epoch(self, epoch):
        # Test model for one epoch
        self.model.eval()
        return self.task.test_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, ckpt, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # Unwrap DDP if needed
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_model(model, ckpt)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    def unwrap_dist_model(self, model):
        # Remove DDP wrapper if distributed; otherwise return as-is
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # Create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # For streaming data (webdataset), no special sampler needed
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # For map-style datasets, may need distributed sampling
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # For certain evaluation, may not use sampler
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)  # Prefetch for speed

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)
            return loader

        loaders = []

        # Create a loader for each dataset/split
        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                # If multiple datasets, create MultiIterLoader
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)
            loaders.append(loader)
        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        # Dictionary of which parameters require gradients
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        # Remove parameters that do not require gradients from state_dict
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model
    
    def _reload_model(self, model,ckpt):
        """
        Load the best checkpoint for evaluation.
        """
        logging.info("Loading checkpoint from {}.".format(ckpt))
        checkpoint = torch.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            # Download checkpoint if it's a URL
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            # Load checkpoint from disk
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        # Log training/validation/test statistics to a file
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass  # (Could implement logging for list stats)

    @main_process
    def log_config(self):
        # Log the experiment configuration to a file
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
