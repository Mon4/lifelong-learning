"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import random
from collections import defaultdict

import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
from torchvision import transforms

from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method


def main():
    args = config.base_parser()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Set the device{device}')

    # Fix the random seeds/ reproducibility
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
    if "autoaug" in args.transforms:
        train_transform.append(select_autoaugment(args.dataset))

    train_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    print(f'Uning train-transforms {train_transform}')

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    print(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    print(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)

    for cur_iter in range(args.n_tasks):
        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        print("[2-1] Prepare a datalist for the current task")
        task_acc = 0.0
        eval_dict = dict()


        # get datalist
        cur_train_datalist = get_train_datalist(args, cur_iter)
        cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)

        print("[2-2] Set environment for the current task")
        method.set_current_dataset(cur_train_datalist, cur_test_datalist)
        # Increment known class for current task iteration.
        method.before_task(cur_train_datalist, cur_iter, args.init_model, args.init_opt)

        # The way to handle streamed samles
        print(f"[2-3] Start to train under {args.stream_env}")
        if args.stream_env == "offline" or args.mode == "joint":
            # Offline Train
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )
            if args.mode == "joint":
                print(f"joint accuracy: {task_acc}")

        elif args.stream_env == "online":
            # Online Train
            print("Train over streamed data once")
            method.train(
                cur_iter=cur_iter,
                n_epoch=1,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )

            method.update_memory(cur_iter)

            # No stremed training data, train with only memory_list
            method.set_current_dataset([], cur_test_datalist)

            # logger.info("Train over memory")
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )

            method.after_task(cur_iter)

        print("[2-4] Update the information for the current task")
        method.after_task(cur_iter)
        task_records["task_acc"].append(task_acc)
        # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
        task_records["cls_acc"].append(eval_dict["cls_acc"])

        # Notify to NSML
        print("[2-5] Report task result")
        print("Metrics/TaskAcc", task_acc, cur_iter)

    # Save file name
    tr_names = ""
    for trans in args.transforms:
        tr_names += "_" + trans
    save_path = f"{args.dataset}/{args.mode}_{args.mem_manage}_{args.stream_env}_msz{args.memory_size}_rnd{args.rnd_seed}{tr_names}"
    np.save(f"results/{save_path}.npy", task_records["task_acc"])

    # Accuracy (A)
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    # Forgetting (F)
    acc_arr = np.array(task_records["cls_acc"])
    # cls_acc = (k, j), acc for j at k
    cls_acc = acc_arr.reshape(-1, args.n_cls_a_task).mean(1).reshape(args.n_tasks, -1)
    for k in range(args.n_tasks):
        forget_k = []
        for j in range(args.n_tasks):
            if j < k:
                forget_k.append(cls_acc[:k, j].max() - cls_acc[k, j])
            else:
                forget_k.append(None)
        task_records["forget"].append(forget_k)
    F_last = np.mean(task_records["forget"][-1][:-1])

    # Intrasigence (I)
    I_last = args.joint_acc - A_last

    print(f"======== Summary =======")
    print(f"A_last {A_last} | A_avg {A_avg} | F_last {F_last} | I_last {I_last}")

if __name__ == "__main__":
    main()
