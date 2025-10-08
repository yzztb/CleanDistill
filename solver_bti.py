
"""ZSKT solvers. solver_tcd_blackFeature"""
"""778-1223为生成smpleset并去ext部分"""
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
from torch import nn
import os
import random
from time import time
import torch.optim as optim
import torch.nn.functional as F
from models import select_model
from utils.loaders import LearnableLoader

from torchvision.utils import make_grid
from utils.trainer_cls import BackdoorModelTrainer
from utils.utils import AverageMeter, AggregateScalar, accuracy
import utils.hypergrad as hg
from utils.config import make_if_not_exist
from datasets import get_test_loader
from utils.syn_vaccine import BackdoorSuspectLoss
from utils.save_load_attack import *

from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

class GeneratedDataset:
    def __init__(self, args, len,ori_shape):
        self.args = args
        self.len = len
        self.ori_shape = ori_shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        dataset_path = "/dataset/ztb/"
        
        file_path = dataset_path + self.args.trigger_pattern + "/" + str(self.args.seed) + "_" + str(idx) + '.npy'
        loaded_numpy_array = np.load(file_path)
        loaded_tensor = torch.from_numpy(loaded_numpy_array).reshape(self.ori_shape)
        return loaded_tensor

    def get_random_sample(self):
        random_index = random.randint(0, self.len - 1)
        return self.__getitem__(random_index)

class GeneratedNoExtremeDataset:
    def __init__(self, args, len,ori_shape):
        self.args = args
        self.len = len
        self.ori_shape = ori_shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        dataset_path = "/dataset/ztb/"
        loaded_numpy_array = np.load(dataset_path+self.args.trigger_pattern+"_noExtreme_v2/"+str(self.args.seed)+"_"+str(idx)+'.npy')

        # 将 NumPy 数组转换为 PyTorch Tensor
        loaded_tensor = torch.from_numpy(loaded_numpy_array).reshape(self.ori_shape)
        return loaded_tensor

    def get_random_sample(self):
        random_index = random.randint(0, self.len - 1)
        return self.__getitem__(random_index)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassifierFromActivation2(nn.Module):
    def __init__(self, wide_resnet_model):
        super().__init__()
        self.block3 = wide_resnet_model.block3
        self.bn1 = wide_resnet_model.bn1
        self.relu = wide_resnet_model.relu
        self.fc = wide_resnet_model.fc
        self.nChannels = wide_resnet_model.nChannels

    def forward(self, x):
        out = self.block3(x)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torchvision.utils import save_image


def normalize_to_cifar(x):
    x = x.clone()
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:  # 防止除以0
        x = (x - x_min) / (x_max - x_min)
    return x


import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def normalize_to_cifar(x):
    x = x.clone()
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    return x


def optimize_input_for_all_classes(
    attack, model, n_classes, image_shape=(3, 32, 32),
    steps=300, lr=0.05, lambda_l2=1e-4,
    device="cuda", seeds=(0, 1, 2, 3, 4), save_dir="./input_ir_analysis"
):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    model.eval()

    all_class_inputs = []  # [n_classes, n_seeds, C, H, W]
    cluster_dists = []

    for cls_idx in range(n_classes):
        seed_inputs = []
        seed_latents = []


        for seed in seeds:
            torch.manual_seed(seed)
            x = torch.rand(image_shape, requires_grad=True, device=device)
            optimizer = torch.optim.Adam([x], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()
                out, *_ = model(x.unsqueeze(0))
                loss = nn.CrossEntropyLoss()(out, torch.tensor([cls_idx], device=device))
                loss = loss + lambda_l2 * torch.norm(x)
                loss.backward()
                optimizer.step()
                x.data.clamp_(0, 1)

            x_norm = normalize_to_cifar(x.detach().cpu())
            seed_inputs.append(x_norm)
            seed_latents.append(x_norm.view(-1))

        seed_inputs = torch.stack(seed_inputs, dim=0)  # [n_seeds, C, H, W]
        all_class_inputs.append(seed_inputs)


        save_path = os.path.join(save_dir, f"{attack}_class_{cls_idx}_optimized_inputs.png")
        save_image(seed_inputs, save_path, nrow=len(seeds), normalize=False)



        seed_latents = torch.stack(seed_latents, dim=0)
        pca = PCA(n_components=3)
        latent_3d = pca.fit_transform(seed_latents.numpy())

        if len(seeds) >= 3:
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
            gmm.fit(latent_3d)
            labels = gmm.predict(latent_3d)
            cluster_sizes = np.bincount(labels)

            if len(cluster_sizes) == 2 and min(cluster_sizes) > 0:
                dist = np.linalg.norm(gmm.means_[0] - gmm.means_[1])
                cluster_dists.append(dist)

                print(f"[!] Class {cls_idx}:dist={dist:.3f}")
        else:
            cluster_dists.append(0.0)


    all_class_inputs = torch.stack(all_class_inputs, dim=0)  # [n_classes, n_seeds, C, H, W]
    grid = all_class_inputs.view(-1, *image_shape) 
    save_image(grid, os.path.join(save_dir, f"{attack}_all_classes_grid.png"),
               nrow=len(seeds), normalize=False)
    print(f"[+] saved: {os.path.join(save_dir, f'{attack}_all_classes_grid.png')}")

    # === 画柱状图 ===
    plt.figure(figsize=(10, 5))
    x = np.arange(n_classes)
    colors = ["#FF4B4B" if i == 0 else "#4B9EFF" for i in range(n_classes)]  
    plt.bar(x, cluster_dists, color=colors)
    plt.xlabel("Class Index")
    plt.ylabel("Cluster Distance (GMM)")
    plt.title("GMM Cluster Center Distances per Class")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{attack}_gmm_cluster_distances.png"))
    plt.close()
    print(f"[+] 保存GMM聚类中心距离柱状图: {os.path.join(save_dir, f'{attack}_gmm_cluster_distances.png')}")

    return all_class_inputs




class ZeroShotKTSolver(object):
    """ Main solver class to train and test the generator and student adversarially """
    
    def __init__(self, args):
        self.args = args
        self.home_path = "/home/ztb/ABD/zskt/ABDfinal/"
        model_path_bench = "/home/ztb/backdoor_zskt/results/normal_training/cifar10/preactresnet18/"+args.trigger_pattern+"/target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1/best_clean_acc.pth"
        ## Student and Teacher Nets
        if args.trigger_pattern in ["bpp","ctrl","ssba","ftrojan","ina","lf","trojan","wanet","refool"]:
            result = load_attack_result(model_path_bench)
            self.result = result
            from models.preactresnet18 import PreActResNet18
            model = PreActResNet18()
            model.load_state_dict(result['model'])
            self.teacher = model.to(args.device)
            
            from torch.utils.data.dataloader import DataLoader
            train_dataloader = DataLoader(self.result['bd_train'], batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                        num_workers=self.args.workers, )
            clean_test_dataloader = DataLoader(self.result['clean_test'], batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                       num_workers=self.args.workers, )
            bd_test_dataloader = DataLoader(self.result['bd_test'], batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                        num_workers=self.args.workers, )
            self.test_loader = clean_test_dataloader
            self.test_poi_loader = bd_test_dataloader
            
        else:
            self.teacher = select_model(dataset=args.dataset,
                                        model_name=args.teacher_architecture,
                                        pretrained=True,
                                        pretrained_models_path=args.pretrained_models_path,
                                        sel_model=args.sel_model,
                                        trigger_pattern=args.trigger_pattern,
                                        ).to(args.device)
            
            
            self.test_loader, self.test_poi_loader = get_test_loader(args)
        self.student = select_model(dataset=args.dataset,
                                    model_name=args.student_architecture,
                                    pretrained=False,
                                    pretrained_models_path=args.pretrained_models_path).to(args.device)
        self.student_gen = select_model(dataset=args.dataset,
                                    model_name=args.student_architecture,
                                    pretrained=False,
                                    pretrained_models_path=args.pretrained_models_path).to(args.device)
        self.student_1 = select_model(dataset=args.dataset,
                                    model_name=args.student_architecture,
                                    pretrained=False,
                                    pretrained_models_path=args.pretrained_models_path).to(args.device)
   
        self.teacher.eval()
        self.student.train()
        self.student_gen.train()
        self.student_1.train()


        ## Loaders
        self.n_repeat_batch = args.n_generator_iter + args.n_student_iter
        self.generator = LearnableLoader(args=args, n_repeat_batch=self.n_repeat_batch).to(device=args.device)

        

        ## Optimizers & Schedulers
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=args.generator_learning_rate)
        self.scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_generator, args.total_n_pseudo_batches, last_epoch=-1)
        self.optimizer_student_gen = optim.Adam(self.student_gen.parameters(), lr=args.student_learning_rate)
        self.scheduler_student_gen = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student_gen, args.total_n_pseudo_batches, last_epoch=-1)


        self.optimizer_student = optim.Adam(self.student.parameters(), lr=args.student_learning_rate)
        self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student, args.total_n_pseudo_batches, last_epoch=-1)
        self.optimizer_student_1 = optim.Adam(self.student_1.parameters(), lr=args.student_learning_rate)
        self.scheduler_student_1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student_1, args.total_n_pseudo_batches, last_epoch=-1)


        ### Set up & Resume
        self.n_pseudo_batches = 0
        if args.sup_backdoor and args.sup_resume is not None:
            self.save_model_path_no_sb = os.path.join(args.save_model_path, args.exp_name_no_sb)
        self.save_model_path = os.path.join(args.save_model_path, args.experiment_name)

        ## Save and Print Args
        print('\n---------')
        make_if_not_exist(self.save_model_path)
        with open(os.path.join(self.save_model_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')

        if self.args.shuf_teacher > 0.:
            self.shuf_loss_fn = BackdoorSuspectLoss(self.teacher, 0. if self.args.pseudo_test_batches > 0 else self.args.shuf_teacher,
                                                    self.args.n_shuf_ens,
                                                    self.args.n_shuf_layer, self.args.no_shuf_grad, device=self.args.device,
                                                    batch_size=self.args.batch_size)
            poi_kl_loss, cl_kl_loss = self.shuf_loss_fn.test_suspect_model(
                self.test_loader, self.test_poi_loader)
            log_file_path =self.home_path +self.args.trigger_pattern+"_"+str(self.args.seed)+"_defense_kl.txt"
            with open(log_file_path, 'w') as f:
                f.write(f'Eval/shuffle_poi_kl_loss:  {poi_kl_loss:.2f}%')
                f.write(f'Eval/shuffle_cl_kl_loss:  {cl_kl_loss:.2f}%')

        else:
            self.shuf_loss_fn = None

        if args.sup_backdoor and args.sup_resume is not None:
            if self.shuf_loss_fn is not None:
                # TODO load shuffled model also.
                # FIXME ad-hoc. We should load the flag from saved shuf_loss_fn.
                print("WARN: Ad-hocly set pseudo_test_flag to be true on resuming for sup.")
                self.shuf_loss_fn.pseudo_test_flag = True
                print("WARN: Ad-hocly set shufl coef to be 0 on resuming for sup.")
                self.shuf_loss_fn.coef = 0.  # if loaded, this will be overwritten.

            self.resume_model(self.save_model_path_no_sb, args.sup_resume, ignore_student_opt=True)

            # reset scheduler.
            self.optimizer_student_gen = optim.Adam(self.student_gen.parameters(), lr=args.sup_resume_lr)
            self.scheduler_student_gen = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_student_gen, args.total_n_pseudo_batches-self.n_pseudo_batches,eta_min=0.0002, last_epoch=-1)
            self.optimizer_student = optim.Adam(self.student.parameters(), lr=args.sup_resume_lr)
            self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_student, args.total_n_pseudo_batches-self.n_pseudo_batches, last_epoch=-1)
            self.optimizer_student_1 = optim.Adam(self.student_1.parameters(), lr=args.sup_resume_lr)
            self.scheduler_student_1 = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_student_1, args.total_n_pseudo_batches-self.n_pseudo_batches, last_epoch=-1)

        if args.resume:
            self.resume_model()



    def run(self):
        epochs = []
        ori_acc = []
        ori_asr = []
        our_acc = []
        our_asr = [] 
        our_acc_noext = []
        our_asr_noext = [] 
        epochs_after=[]
        ori_acc_after=[]
        ori_asr_after=[]
        target_class_list=[0]
        self.teacher.eval()

        # 获取 activation2 的输出形状
        with torch.no_grad():
            teacher_logits, activation1, activation2, *teacher_activations = self.teacher(torch.randn(1, 3, 32, 32).to(device=self.args.device))
        ir_shape = activation2.shape[1:]  # 去掉 batch 维度

        ir_dict = optimize_input_for_all_classes(
            attack=self.args.trigger_pattern,
            model=self.teacher,
            n_classes=10,
            image_shape=(3, 32, 32),
            steps=300,
            lr=0.05,
            lambda_l2=1e-4,
            device=self.args.device,
            seeds=list(range(15)),  # 用8个seed提高鲁棒性
            save_dir=os.path.join(self.home_path, "input_ir_results")
)

        
        # fname = "/home/ztb/ABD/zskt/generator/" + self.args.trigger_pattern + "_generator.pth"
        # self.generator.load_state_dict(torch.load(fname)['state_dict'])
        
        # sample_set_size = 400
        # ext_value = 0.8
        # extreme_limit = 0.95
        # n_classes = 10
        # target_per_class = sample_set_size

        # activation_dict = defaultdict(list)
        # class_counter = defaultdict(int)
        # extreme_counter = 0  # 统计超过 extreme_limit 的样本数
        # extreme_class_counter = defaultdict(int)  # 统计每个类超过 extreme_limit 的次数

        # print("generating balanced training data...")
        # count = 0

        # with torch.no_grad():
        #     while True:
        #         if count % 500 == 0:
        #             print(f"generated batches: {count}")

        #         x_pseudo = next(self.generator)
        #         teacher_logits, activation1, activation2, *teacher_activations = self.teacher(x_pseudo)

        #         probs = torch.softmax(teacher_logits, dim=1)
        #         teacher_maxes, teacher_argmaxes = torch.max(probs, dim=1)

        #         mask = teacher_maxes >= ext_value
        #         if mask.any():
        #             selected_indices = torch.nonzero(mask, as_tuple=True)[0]
        #             activation2_flat = activation2.view(activation2.size(0), -1)

        #             for idx in selected_indices:
        #                 class_idx = teacher_argmaxes[idx].item()

        #                 # 统计 extreme_limit 的数量
        #                 if teacher_maxes[idx] >= extreme_limit:
        #                     extreme_counter += 1
        #                     extreme_class_counter[class_idx] += 1

        #                 # 收集样本直到每类达到 target_per_class
        #                 if class_counter[class_idx] < target_per_class:
        #                     activation_dict[class_idx].append(activation2_flat[idx].cpu())
        #                     class_counter[class_idx] += 1

        #         count += 1

        #         # 检查是否所有类别都达标
        #         if all(v >= target_per_class for v in class_counter.values()):
        #             print("[+] 已收集到每类样本数量目标，停止生成")
        #             break
        
        # extreme_class_counter = sorted(extreme_class_counter.items(), key=lambda x: x[1], reverse=True)
        # print(f"总共生成超过 {extreme_limit} 的样本数: {extreme_counter}")
        # for cls_idx, cnt in dict(extreme_class_counter).items():
        #     print(f"Class {cls_idx}: 超过 {extreme_limit} 的次数 = {cnt}")

            #---------------------------generate biased number of samples for classes
            # while count < sample_set_size:
            #     if count % 500 == 0:
            #         print(f"generated samples: {count}")

            #     # 生成一批伪样本
            #     x_pseudo = next(self.generator)
            #     teacher_logits, activation1, activation2, *teacher_activations = self.teacher(x_pseudo)

            #     # 计算 softmax 概率、最大概率和对应类别
            #     probs = torch.softmax(teacher_logits, dim=1)
            #     teacher_maxes, teacher_argmaxes = torch.max(probs, dim=1)

            #     # 筛选概率 >= ext_value 的样本
            #     mask = teacher_maxes >= ext_value
            #     if mask.any():
            #         selected_indices = torch.nonzero(mask, as_tuple=True)[0]

            #         # 展平 activation2 方便后续GMM拟合
            #         activation2_flat = activation2.view(activation2.size(0), -1)

            #         for idx in selected_indices:
            #             class_idx = teacher_argmaxes[idx].item()
            #             activation_dict[class_idx].append(activation2_flat[idx].cpu())

            #     count += 1

        # ---------- 用GMM判断每个类别的分布模式 ----------
        # for class_idx, act_list in activation_dict.items():
        #     if not act_list:
        #         continue

        #     result = self.check_single_or_mixture(act_list, visualize=True, class_idx=class_idx)
        #     print(f"Class {class_idx}: {result}")


        # ---------- 用class中样本数量进行筛选 ----------
        # class_counts = {cls: len(acts) for cls, acts in activation_dict.items()}

        # # 按样本数量排序，取前5个
        # sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        # topk_classes = [cls for cls, count in sorted_classes[:k]]

        # print("=== 样本数量统计 ===")
        # for cls, count in sorted_classes:
        #     print(f"Class {cls}: {count} samples")
        # print(f"[+] 选择数量最多的k个类别: {topk_classes}")

        # # 构建新的 dict 只保留 top5
        # activation_dict_topk = {cls: activation_dict[cls] for cls in topk_classes}
        # ir_dict_topk = {cls: ir_dict[cls] for cls in topk_classes}

        # self.compute_distance_sum(activation_dict_topk,ir_dict=ir_dict_topk)
        # self.compute_distance_sum(activation_dict,ir_dict=ir_dict)

        return 100



    def compute_distance_sum(self, activation_dict, ir_dict, reduction='mean'):
        """
        方法一：计算每类样本到对应IR的距离总和或均值。

        参数:
            activation_dict: {class_idx: [tensor, tensor, ...]} 每类的样本激活
            ir_dict: {class_idx: tensor} 生成的IR
            reduction: 'sum' 或 'mean'
        返回:
            dist_per_class: {class_idx: 距离总和或均值}
        """
        dist_per_class = {}
        for cls_idx, activations in activation_dict.items():
            ir_vec = ir_dict[cls_idx].flatten().to(activations[0].device)
            dists = [torch.norm(act.flatten() - ir_vec).item() for act in activations]
            if reduction == 'sum':
                dist_per_class[cls_idx] = sum(dists)
            elif reduction == 'mean':
                dist_per_class[cls_idx] = sum(dists) / len(dists)
            else:
                raise ValueError("reduction must be 'sum' or 'mean'")

        # 打印结果并找最大
        print("=== 每类距离统计 ===")
        dist_per_class = dict(sorted(dist_per_class.items(), key=lambda x: x[1], reverse=True))
        for k, v in dist_per_class.items():
            print(f"Class {k}: {v:.4f}")

        max_class = max(dist_per_class, key=dist_per_class.get)
        print(f"[+] 距离最大的类别: {max_class}, 距离={dist_per_class[max_class]:.4f}")
        return dist_per_class, max_class



    def check_single_or_mixture(self, activation_list, visualize=False, class_idx=None):
        """
        activation_list: list of torch.Tensor, 每个是 activation2 展平后的向量
        visualize: 是否画PCA降维的可视化
        return: "single" 或 "mixture"
        """
        # 拼成矩阵 [N, D]
        X = torch.stack(activation_list).cpu().numpy()
        
        # --- 降维以提高GMM稳定性 ---
        pca_dim = min(10, X.shape[1])  # 保留最多10维
        pca = PCA(n_components=pca_dim, random_state=0)
        X_pca = pca.fit_transform(X)

        # GMM 拟合
        gmm1 = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(X_pca)
        gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(X_pca)
        
        bic1 = gmm1.bic(X_pca)
        bic2 = gmm2.bic(X_pca)

        print(f"[Class {class_idx}] BIC (1 component): {bic1:.2f}")
        print(f"[Class {class_idx}] BIC (2 components): {bic2:.2f}")

        result = "mixture" if bic2 + 1200 < bic1 else "single"

        # --- 可视化降维后的分布 ---
        if visualize:
            if X_pca.shape[1] > 2:
                # 降到2D方便画散点
                X_vis = PCA(n_components=2, random_state=0).fit_transform(X_pca)
            else:
                X_vis = X_pca

            plt.figure()
            if result == "mixture":
                labels = gmm2.predict(X_pca)
                plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap="coolwarm", alpha=0.6)
            else:
                plt.scatter(X_vis[:, 0], X_vis[:, 1], color='blue', alpha=0.6)
            
            plt.title(f"Class {class_idx} ({result})")
            plt.xlabel("PCA-1")
            plt.ylabel("PCA-2")
            plt.grid(True)
            plt.show()
            output_path = self.home_path+self.args.trigger_pattern+"_"+str(self.args.seed)+"_"+str(class_idx)+"_distribution.svg" 
            plt.savefig(output_path,format='svg')

        return result



    def get_state(self, model, data_loader):

            model.eval()
            pred_position_counts = {}
            # index: the final prediction class 
            # i: the i-th outcome from max to min
            #value: the time that this index class apeear as the i-th outcome 
            for index in range(10):
                pred_position_counts[index] = 0
            num_data = 0
            mean_logits = torch.zeros(10,)
            with torch.no_grad():
                
                for x in data_loader:
                    num_data += 1
                    # logits = model(x)
                    # _, max_indice = torch.max(logits, dim=1)
                    # x, y = x.to(self.args.device), max_indice.to(self.args.device)
                    for i in range(x.size(0)):  # 按照第一个维度遍历
                        reshaped_tensor = x[i].view(128, 3, 32,32)
                        reshaped_tensor= reshaped_tensor.to(self.args.device)
                        student_logits, *student_activations = model(reshaped_tensor)
                        #_, target = torch.topk(y, k=10, dim=1)
                        data = student_logits.data
                        
                        mean_logits +=data.mean(dim=0).cpu()

                        _, pred = torch.topk(student_logits.data, k=10, dim=1)

                        # for pos, index in enumerate(target):
                        #     index = index.item()
                        #     true_position_counts[index][pos] += 1
                        for each_result in torch.unbind(pred, dim=0):
                            for pos, index in enumerate(each_result):
                                index = index.item()
                                pred_position_counts[index] += pos
            # model.train()  # do not change the mode of teacher.

            return  mean_logits, pred_position_counts
    def get_result(self, model, data_loader):

            model.eval()
            logits_sum = torch.zeros(10, device=self.args.device)
            with torch.no_grad():
                for x in data_loader:
                    for i in range(x.size(0)):  # 按照第一个维度遍历
                        reshaped_tensor = x[i].view(128, 3, 32,32)
                        reshaped_tensor= reshaped_tensor.to(self.args.device)
                        student_logits, *student_activations = model(reshaped_tensor)
                        logits_sum += student_logits.sum(dim=0)
                return  logits_sum
    
    def test(self, model, data_loader):

        model.eval()
        running_test_acc = AggregateScalar()

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.args.device), y.to(self.args.device)
                student_logits, *student_activations = model(x)
                acc = accuracy(student_logits.data, y, topk=(1,))[0]
                running_test_acc.update(float(acc), x.shape[0])

        # model.train()  # do not change the mode of teacher.
        return running_test_acc.avg()
    
    def test_with_class(self, model, data_loader):
        model.eval()
    
        num_classes = 10  # Number of classes
        correct_per_class = torch.zeros(num_classes, device=self.args.device)
        total_per_class = torch.zeros(num_classes, device=self.args.device)
        running_test_acc = AggregateScalar()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.args.device), y.to(self.args.device)
                student_logits, *student_activations = model(x)
                acc = accuracy(student_logits.data, y, topk=(1,))[0]
                running_test_acc.update(float(acc), x.shape[0])

                # Get predicted classes
                _, predicted = student_logits.max(1)

                # Update the count for each class
                for i in range(num_classes):
                    total_per_class[i] += (y == i).sum().item()
                    correct_per_class[i] += ((predicted == i) & (y == i)).sum().item()

        # Calculate the accuracy per class
        per_class_acc = correct_per_class / total_per_class
        per_class_acc[total_per_class == 0] = 0.0  # Handle cases where there are no examples for a class

        return per_class_acc.cpu().numpy() , running_test_acc.avg()


    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()


    def divergence(self, student_logits, teacher_logits, reduction='mean'):
        divergence = F.kl_div(F.log_softmax(student_logits / self.args.KL_temperature, dim=1), F.softmax(teacher_logits / self.args.KL_temperature, dim=1), reduction=reduction)  # forward KL

        return divergence


    def KT_loss_generator(self, student_logits, teacher_logits, reduction='mean'):

        divergence_loss = self.divergence(student_logits, teacher_logits, reduction=reduction)
        total_loss = - divergence_loss

        return total_loss


    def KT_loss_student(self, student_logits, student_activations, teacher_logits, teacher_activations):

        divergence_loss = self.divergence(student_logits, teacher_logits)
        if self.args.AT_beta > 0:
            at_loss = 0
            for i in range(len(student_activations)):
                at_loss = at_loss + self.args.AT_beta * self.attention_diff(student_activations[i], teacher_activations[i])
        else:
            at_loss = 0

        total_loss = divergence_loss + at_loss

        return total_loss
    
    def resume_model(self, save_path=None, save_name="last.pth", ignore_student_opt=False):
        if save_path is None:
            save_path = self.save_model_path
        checkpoint_path = os.path.join(save_path, save_name)
        if os.path.isfile(checkpoint_path):
            print(f"Load checkpoint <= {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            print(' Resuming at batch iter {} with'.format(checkpoint['n_pseudo_batches']))
            print(f" Acc {checkpoint['Eval/test_acc']}")
            if 'Eval/test_poi_acc' in checkpoint:
                print(f" ASR {checkpoint['Eval/test_poi_acc']}")
            print('Running an extra {} iterations'.format(self.args.total_n_pseudo_batches - checkpoint['n_pseudo_batches']))
            self.n_pseudo_batches = checkpoint['n_pseudo_batches']
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
            self.scheduler_generator.load_state_dict(checkpoint['scheduler_generator'])
            self.student_gen.load_state_dict(checkpoint['student_state_dict'])
            if not ignore_student_opt:
                self.optimizer_student_gen.load_state_dict(checkpoint['optimizer_student'])
                self.scheduler_student_gen.load_state_dict(checkpoint['scheduler_student'])
            if 'shuf_loss_fn' in checkpoint:
                self.shuf_loss_fn.load_state_dict(checkpoint['shuf_loss_fn'])
        else:
            print(os.path.abspath(checkpoint_path))
            raise FileNotFoundError(f"Not found checkpoint at {checkpoint_path}")

    def save_model(self, info, file_name=None):
        make_if_not_exist(self.save_model_path)
        save_dict = {'args': self.args,
                    'n_pseudo_batches': self.n_pseudo_batches,
                    'generator_state_dict': self.generator.state_dict(),
                    'student_state_dict': self.student_gen.state_dict(),
                    'optimizer_generator': self.optimizer_generator.state_dict(),
                    'optimizer_student': self.optimizer_student_gen.state_dict(),
                    'scheduler_generator': self.scheduler_generator.state_dict(),
                    'scheduler_student': self.scheduler_student_gen.state_dict(),
                    # self.shuf_loss_fn.pseudo_dataset.
                    **info}
        
        if self.shuf_loss_fn is not None:
            save_dict['shuf_loss_fn'] = self.shuf_loss_fn.state_dict()
        if file_name is not None:
            fname = os.path.join(self.save_model_path, file_name)
            torch.save(save_dict, fname)
            print(f"Save model => {fname}")
        fname = os.path.join(self.save_model_path, 'last.pth')
        torch.save(save_dict, fname)
        print(f"Save model => {fname}")

    def save_generator(self):
        fname = "/home/ztb/ABD/zskt/generator/"+self.args.trigger_pattern+"_"+'generator.pth'
        torch.save({'state_dict': self.generator.state_dict()}, fname)
        print(f"Save generator to {fname}")
    
    def save_pseudo_images(self, i_iter, images, **kwargs):
        img_folder = self.get_pseudo_images_folder()
        fname = os.path.join(img_folder, f'{i_iter}.pth')
        make_if_not_exist(img_folder)
        torch.save({'imgs': images.data.cpu(), **kwargs}, fname)
        # print(f"Save images to {fname}")

    def get_pseudo_images_folder(self):
        img_folder = os.path.join(self.save_image_path, 'images')
        return img_folder
    
    