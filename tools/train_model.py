import os
import time
import torch
import shutil
import argparse
from pprint import pprint
from datasets.dataset import build_dataset
from models.model import build_model
from models.loss import build_loss
from torch.utils.tensorboard import SummaryWriter
from utils.utils import set_seed, parse_yaml_config, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="simple self-supervised learning")
    parser.add_argument("--config", help="the path of config file")

    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.init_trainer(cfg)

    def init_trainer(self, cfg):
        # create dir for tensorboard and model-saving
        base_save_dir = os.path.join(cfg["train"]["base_save_dir"], cfg["train"]["tag"])
        tensorboard_dir = os.path.join(base_save_dir, "tensorboard")
        self.model_save_dir = os.path.join(base_save_dir, "model")
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.tf_writer = SummaryWriter(tensorboard_dir)

        config_file_name = os.path.basename(args.config)
        shutil.copy2(args.config, os.path.join(base_save_dir, config_file_name))

        # build_model
        model_name = cfg["model"]["name"]
        model = build_model(model_name, cfg["model"])

        model_param = model.parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        loss_name = cfg["train"]["loss"]["name"]
        cfg["train"]["loss"]["device"] = self.device
        self.loss_func = build_loss(loss_name, cfg["train"]["loss"])

        self.optimizer = torch.optim.Adam(model_param, lr=cfg["train"]["lr"])

        # build train dataset
        train_dataset = build_dataset(cfg["train"]["dataset"], cfg["train"])

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=True,
        )

        # build validation dataset
        self.val_loader = None
        if "val" in cfg.keys():
            val_dataset = None
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg["val"]["batch_size"],
                shuffle=True,
                num_workers=cfg["val"]["num_workers"],
                pin_memory=True,
            )

    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()
        end = time.time()

        for idx, input_data in enumerate(self.train_loader):
            x1 = input_data[0].to(self.device)
            x2 = input_data[1].to(self.device)

            input_list = (x1, x2)
            output = self.model(input_list)

            loss = self.loss_func(output)
            losses.update(loss.item(), x1.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            self.num_count_iter += 1
            if idx % self.num_iter_log == 0:
                self.tf_writer.add_scalar("loss", losses.val, self.num_count_iter)

                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                        epoch,
                        idx,
                        len(self.train_loader),
                        batch_time=batch_time,
                        loss=losses,
                    )
                )

        self.batch_time_avg.append(batch_time.avg)
        self.batch_time_sum += batch_time.sum

        print(f"Total batch time: {self.batch_time_sum}\n")

    def val_after_epoch(self, epoch):
        return None

    def train(self):
        self.batch_time_avg = list()
        self.batch_time_sum = 0
        self.num_iter_log = self.cfg["train"]["num_iter_log"]
        num_epoch_save_model = self.cfg["train"]["num_epoch_save_model"]

        total_epoch = self.cfg["train"]["total_epoch"]
        self.num_count_iter = -1

        for epoch in range(1, total_epoch + 1):
            # train
            self.train_one_epoch(epoch)

            # valid
            if (
                self.val_loader is not None
                and epoch % (self.cfg["val"]["num_epoch_interval"]) == 0
            ):
                self.val_after_epoch(epoch)

            # save model
            if (epoch % num_epoch_save_model == 0) or (epoch == total_epoch):
                save_model_path = os.path.join(
                    self.model_save_dir, "model_" + str(epoch) + ".pth"
                )

                torch.save(self.model.state_dict(), save_model_path)

        self.tf_writer.flush()
        print("well done")


if __name__ == "__main__":
    args = parse_args()
    cfg = parse_yaml_config(args.config)
    print("the config is ")
    pprint(cfg)
    print("------")

    # random seed
    set_seed(3)

    trainer = Trainer(cfg)
    trainer.train()
