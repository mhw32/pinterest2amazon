"""
General agent class for training local aggregation models.
"""

import os
import sys
import copy
import json
import pickle
import logging
import numpy as np
from tqdm import tqdm
from itertools import product, chain
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.utils.utils import \
    save_checkpoint as save_snapshot, \
    copy_checkpoint as copy_snapshot, \
    AverageMeter
from src.utils.setup import print_cuda_statistics
from src.objectives.localagg import \
    LocalAggregationLoss, MemoryBank, Kmeans
from src.objectives.instance import InstanceDiscriminationLoss
from src.datasets.datasets import load_datasets


class BaseAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        
        self._set_seed()  # set seed as early as possible
        
        self._load_datasets()
        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset, shuffle=True)
        if self.config.validate:
            self.val_loader, self.val_len = self._create_dataloader(self.val_dataset, shuffle=False)

        self._choose_device()
        self._create_model()
        self._create_optimizer()

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0
        
        # we need these to decide best loss
        self.current_loss = 0
        self.current_val_metric = 0
        self.best_val_metric = 0
        self.iter_with_no_improv = 0

        try:  # hack to handle different versions of TensorboardX
            self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)
        except:
            self.summary_writer = SummaryWriter(logdir=self.config.summary_dir)

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda: torch.cuda.manual_seed(self.manual_seed)

        if self.cuda:
            if not isinstance(self.config.gpu_device, list):
                self.config.gpu_device = [self.config.gpu_device]
            num_gpus = len(self.config.gpu_device)
            self.multigpu = num_gpus > 1 and torch.cuda.device_count() > 1

            if not self.multigpu:  # e.g. just 1 GPU
                gpu_device = self.config.gpu_device[0]
                self.logger.info("User specified 1 GPU: {}".format(gpu_device))
                self.device = torch.device("cuda")
                torch.cuda.set_device(gpu_device)
            else:
                gpu_devices = ','.join(self.config.gpu_device)
                self.logger.info("User specified {} GPUs: {}".format(
                    num_gpus, gpu_devices))
                # if we use multiple gpus..., no nice way of specifying
                # such, so we will instead hack the environment variable
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
                self.device = torch.device("cuda")
            
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _create_dataloader(self, dataset, shuffle=True):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=self.config.optim_params.batch_size, 
                            shuffle=shuffle, pin_memory=True, 
                            num_workers=self.config.data_loader_workers)
        
        return loader, dataset_size

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
            self.cleanup()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            self.cleanup()
            raise e

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            if (self.config.validate and 
                epoch % self.config.optim_params.validate_freq == 0):
                self.validate()  # validate every now and then
            self.save_checkpoint()
            
            # check if we should quit early bc bad perf
            if self.iter_with_no_improv > self.config.optim_params.patience:
                self.logger.info("Exceeded patience. Stop training...") 
                break

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        self.summary_writer.export_scalars_to_json(
            os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')

    def cleanup(self):
        """
        Undo any global changes that the Agent may have made
        """
        if self.multigpu:
            del os.environ['CUDA_VISIBLE_DEVICES']


class TrainAgent(BaseAgent):
    """
    Primary agent for training unimodal <image> models. We also 
    assume a second modality exists (labels) so we can score 
    performance that way. 

    @param config: DotMap
                   configuration settings
    """
    def __init__(self, config):
        super(TrainAgent, self).__init__(config)
        # initialize objects specific to local aggregation
        self._init_memory_bank()
        self.cluster_labels = None
        if self.config.loss_params.loss == 'LocalAggregationLoss':
            self._init_cluster_labels()
            self.km = None  # will be populated by a kmeans model
            # if user did not specify kmeans_freq, then set to constant
            if self.config.loss_params.kmeans_freq is None:
                self.config.loss_params.kmeans_freq = (
                    len(self.train_dataset) // 
                    self.config.optim_params.batch_size)

        self.val_acc = []
        self.train_loss = []
        self.train_extra = []

    def _init_memory_bank(self, attr_name='memory_bank'):
        data_len = len(self.train_dataset)
        memory_bank = MemoryBank(
            data_len, self.config.model_params.out_dim, self.device)
        setattr(self, attr_name, memory_bank)

    def load_memory_bank(self, memory_bank):
        self._load_memory_bank(memory_bank, attr_name='memory_bank')

    def _load_memory_bank(self, memory_bank, attr_name='memory_bank'):
        # if we want to load an existing memory bank
        memory_bank = copy.deepcopy(memory_bank)
        memory_bank.device = self.device
        memory_bank._bank = memory_bank._bank.cpu()
        memory_bank._bank = memory_bank._bank.to(self.device)
        setattr(self, attr_name, memory_bank)

    def get_memory_bank(self):
        return self._get_memory_bank(attr_name='memory_bank')

    def _get_memory_bank(self, attr_name='memory_bank'):
        return getattr(self, attr_name)

    def _init_cluster_labels(self, attr_name='cluster_labels'):
        no_kmeans_k = self.config.loss_params.n_kmeans  # how many wil be train
        data_len = len(self.train_dataset)
        # initialize cluster labels
        cluster_labels = torch.arange(data_len).long()
        cluster_labels = cluster_labels.unsqueeze(0).repeat(no_kmeans_k, 1)
        setattr(self, attr_name, cluster_labels)

    def _load_image_transforms(self):
        image_size = self.config.data_params.image_size
        if self.config.data_params.image_augment:
            train_transforms = transforms.Compose([
                # these are borrowed from 
                # https://github.com/zhirongw/lemniscate.pytorch/blob/master/main.py
                transforms.RandomResizedCrop(image_size, scale=(0.2,1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(256),  # FIXME: hardcoded for 224 image size
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(256),  # FIXME: hardcoded for 224 image size
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
            test_transforms = copy.copy(train_transforms)
        
        return train_transforms, test_transforms

    def _load_datasets(self):
        train_transforms, test_transforms = self._load_image_transforms()

        # build training dataset
        self.train_dataset = load_datasets(self.config.data_params.name, split='train', 
                                           image_transforms=train_transforms)
        if self.config.validate:
            # build validation set
            self.val_dataset = load_datasets(self.config.data_params.name, split='validation', 
                                             image_transforms=test_transforms)
        
        # save some stuff to config
        self.config.data_params.n_channels = 3
        train_samples = self.train_dataset.dataset.samples
        train_labels = [train_samples[i][1] for i in range(len(train_samples))]
        self.train_ordered_labels = np.array(train_labels)

    def _create_model(self):
        assert self.config.data_params.image_size == 224
        resnet_class = getattr(models, self.config.model_params.resnet_version)
        model = resnet_class(pretrained=False, 
                             num_classes=self.config.model_params.out_dim)
        model = model.to(self.device)
        if self.multigpu:
            # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
            # i think this is all you have to do to make things parallelized in PyTorch
            model = nn.DataParallel(model)
        self.model = model

    def _set_models_to_eval(self):
        self.model = self.model.eval()

    def _set_models_to_train(self):
        self.model = self.model.train()

    def _create_optimizer(self): 
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.config.optim_params.learning_rate,
                                     momentum=self.config.optim_params.momentum,
                                     weight_decay=self.config.optim_params.weight_decay)

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch))

        self._set_models_to_train()  # turn on train mode
        epoch_loss = AverageMeter()

        for batch_i, (indices, images, labels) in enumerate(self.train_loader):
            batch_size = images.size(0)

            # cast elements to CUDA
            indices = indices.to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)

            # do a forward pass
            outputs = self.model(images)

            # define custom class to compute loss
            loss_class = globals()[self.config.loss_params.loss]  # toggle between ID and LA
            loss_fn = loss_class(indices, outputs, self.memory_bank,
                                 k=self.config.loss_params.k,
                                 t=self.config.loss_params.t,
                                 m=self.config.loss_params.m)

            # compute the loss
            loss, extra = loss_fn.get_loss(self.cluster_labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # since this is training, we need to update the memory bank
            with torch.no_grad():
                new_data_memory = loss_fn.updated_new_data_memory()
                self.memory_bank.update(indices, new_data_memory)

                # we only need to train kmeans for local aggregation
                if (self.config.loss_params.loss == 'LocalAggregationLoss' and
                    (self.current_iteration % self.config.loss_params.kmeans_freq == 0 or batch_i == 0)):
                    self.logger.info('Fitting K-means with FAISS')
                    # get kmeans clustering (update our saved clustering)
                    k = [self.config.loss_params.kmeans_k for _ in
                         range(self.config.loss_params.n_kmeans)]
                    # NOTE: we use a different gpu for FAISS otherwise cannot fit onto memory
                    self.km = Kmeans(k, self.memory_bank, gpu_device=self.config.faiss_gpu_device)
                    self.cluster_labels = self.km.compute_clusters()

            epoch_loss.update(loss.item(), batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg})

            self.summary_writer.add_scalars("epoch/loss", {'loss': epoch_loss.val}, 
                                            self.current_iteration)
            if len(extra) > 0:  # save any extra numbers we want
                extra_names = ['extra:{}'.format(i) for i in range(len(extra))]
                self.summary_writer.add_scalars("epoch/extra",
                                                dict(zip(extra_names, extra)),
                                                self.current_iteration)
            
            self.train_loss.append(epoch_loss.val)
            self.train_extra.append(extra)

            self.current_iteration += 1
            tqdm_batch.update()

        self.current_loss = epoch_loss.avg
        tqdm_batch.close()
   
    def validate(self):
        # For validation, for each image, we find the closest neighbor in the 
        # memory bank (training data), take its class! We compute the accuracy.

        num_batches = self.val_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Val]")

        self._set_models_to_eval()
        num_correct = 0.
        num_total = 0.

        with torch.no_grad():
            for _, images, labels in self.val_loader:
                batch_size = images.size(0)

                # cast elements to CUDA
                images = images.to(self.device)
                outputs = self.model(images)

                # use memory bank to ge the top 1 neighbor
                # from the training dataset
                all_dps = self.memory_bank.get_all_dot_products(outputs)
                _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
                neighbor_idxs = neighbor_idxs.squeeze(1)  # shape: batch_size
                neighbor_idxs = neighbor_idxs.cpu().numpy()  # convert to numpy
                # fetch the actual label of each example
                neighbor_labels = self.train_ordered_labels[neighbor_idxs]
                neighbor_labels = torch.from_numpy(neighbor_labels).long()

                num_correct += torch.sum(neighbor_labels == labels).item()
                num_total += batch_size

                tqdm_batch.set_postfix({"Val Accuracy": num_correct / num_total})
                tqdm_batch.update()
      
        self.summary_writer.add_scalars("Val/accuracy", 
                                        {'accuracy': num_correct / num_total}, 
                                        self.current_val_iteration)

        self.current_val_iteration += 1
        self.current_val_metric = num_correct / num_total
        
        # save if this was the best validation accuracy
        # (important for model checkpointing)
        if self.current_val_metric >= self.best_val_metric:  # NOTE: >= for accuracy
            self.best_val_metric = self.current_val_metric
            self.iter_with_no_improv = 0   # reset patience
        else:
            self.iter_with_no_improv += 1  # no improvement 
        
        tqdm_batch.close()
      
        # store the validation metric from every epoch
        self.val_acc.append(self.current_val_metric)

        return self.current_val_metric

    def load_checkpoint(self, filename, checkpoint_dir=None, 
                        load_memory_bank=True, load_model=True, 
                        load_optim=True, load_epoch=True):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=self.device)

            if load_epoch:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            if load_memory_bank: # load memory_bank
                self._load_memory_bank(checkpoint['memory_bank'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e

    def copy_checkpoint(self, filename="checkpoint.pth.tar"):
        if self.current_epoch % self.config.copy_checkpoint_freq == 0:
            copy_snapshot(  
                filename=filename, folder=self.config.checkpoint_dir,
                copyname='checkpoint_epoch{}.pth.tar'.format(self.current_epoch),
            )

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'memory_bank': self.memory_bank,
            'cluster_labels': self.cluster_labels,
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'config': self.config,
            'val_acc': np.array(self.val_acc),
            'train_loss': np.array(self.train_loss),
            'train_extra': np.array(self.train_extra),
        }

        # if we aren't validating, then every time we save is the 
        # best new epoch!
        is_best = ((self.current_val_metric == self.best_val_metric) or 
                   not self.config.validate)
        save_snapshot(out_dict, is_best, filename=filename,
                      folder=self.config.checkpoint_dir)
        self.copy_checkpoint()
