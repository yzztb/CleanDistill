"""ZSKT solvers. _newGenStu_final_stable_gen(quantile ext_value used for each class individually) 333-384regenerate DB without extreme values testing extlimit(0.95, 0.9, 0.85)"""
import numpy as np
import torch
from torch import nn
import os
import random
from time import time
import torch.optim as optim
import torch.nn.functional as F
from models import select_model
from utils.loaders_new import LearnableLoader_new

from torchvision.utils import make_grid
from utils.trainer_cls import BackdoorModelTrainer
from utils.utils import AverageMeter, AggregateScalar, accuracy
import utils.hypergrad as hg
from utils.config import make_if_not_exist
from datasets import get_test_loader
from utils.syn_vaccine import BackdoorSuspectLoss
from utils.save_load_attack import *

class GeneratedDataset:
    def __init__(self, args, len,ori_shape):
        self.args = args
        self.len = len
        self.ori_shape = ori_shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        dataset_path = "/dataset/ztb/"
        # while idx<self.len:
        #     file_path = dataset_path + self.args.trigger_pattern + "/" + str(self.args.seed) + "_" + str(idx) + '.npy'

        #     try:
        #         # 加载 NumPy 数组
        #         loaded_numpy_array = np.load(file_path)
        #         # 尝试将 NumPy 数组转换为 PyTorch Tensor，并重新调整为 ori_shape
        #         loaded_tensor = torch.from_numpy(loaded_numpy_array).reshape(self.ori_shape)
        #         break  # 如果成功加载且没有异常，跳出循环
        #     except ValueError as e:
        #         # 打印错误信息和文件路径
        #         print(f"ValueError occurred: {e}. The file path is: {file_path}. Retrying with idx + 1...")
        #         # idx 加 1 并继续循环
        #         idx += 1
        #     except Exception as e:
        #         # 捕获其他异常并显示文件路径
        #         print(f"An error occurred: {e}. The file path is: {file_path}")
        #         raise e  # 重新抛出其他异常
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
        loaded_numpy_array = np.load(dataset_path+self.args.trigger_pattern+"_noExtreme/"+str(self.args.seed)+"_"+str(idx)+'.npy')

        # 将 NumPy 数组转换为 PyTorch Tensor
        loaded_tensor = torch.from_numpy(loaded_numpy_array).reshape(self.ori_shape)
        return loaded_tensor

    def get_random_sample(self):
        random_index = random.randint(0, self.len - 1)
        return self.__getitem__(random_index)

class ZeroShotKTSolver(object):
    """ Main solver class to train and test the generator and student adversarially """
    
    def __init__(self, args):
        self.args = args
        self.home_path = "/home/ztb/ABD/zskt/ABDfinal/"
        model_path_bench = "/home/ztb/backdoor_zskt/results/normal_training/"+self.args.dataset.lower()+"/"+args.teacher_architecture+"/"+args.trigger_pattern+"/target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1/"+"best_clean_acc.pth"
        ## Student and Teacher Nets
        if args.trigger_pattern in ["bpp","ctrl","ssba","ftrojan","ina","lf","trojan","wanet","refool","badnet_sq","blend","sig"]:
            result = load_attack_result(model_path_bench)
            self.result = result
            from models.preactresnet18 import PreActResNet18
            model = select_model(dataset=args.dataset,
                                        model_name=args.teacher_architecture,
                                        pretrained=True,
                                        pretrained_models_path=args.pretrained_models_path,
                                        sel_model=args.sel_model,
                                        trigger_pattern=args.trigger_pattern,
                                        )
                
            model.load_state_dict(result['model'])
            self.teacher = model.to(args.device)
            self.teacher_gen = copy.deepcopy(model).to(args.device)
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
            
            self.teacher_gen = select_model(dataset=args.dataset,
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
        self.student_gen_1 = select_model(dataset=args.dataset,
                                    model_name=args.student_architecture,
                                    pretrained=False,
                                    pretrained_models_path=args.pretrained_models_path).to(args.device)
        self.teacher.eval()
        self.student.train()
        self.teacher_gen.eval()
        self.student_gen.train()
        self.student_1.train()
        self.student_gen_1.train()

        ## Loaders
        self.n_repeat_batch = args.n_generator_iter + args.n_student_iter
        self.generator = LearnableLoader_new(args=args, n_repeat_batch=self.n_repeat_batch).to(device=args.device)

        

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
                self.optimizer_student_gen, args.total_n_pseudo_batches-self.n_pseudo_batches, last_epoch=-1)
            self.optimizer_student = optim.Adam(self.student.parameters(), lr=args.sup_resume_lr)
            self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_student, args.total_n_pseudo_batches-self.n_pseudo_batches, last_epoch=-1)

        if args.resume:
            self.resume_model()

    def run(self):
        #self.args.total_n_pseudo_batches = 400
        epochs = []
        ori_acc = []
        ori_asr = []
        our_acc = []
        our_asr = [] 
        our_acc_noext = []
        our_asr_noext = [] 
        # if (self.n_pseudo_batches+1) // self.args.log_freq == 1:
        if self.args.trigger_pattern in ["bpp","ctrl","ssba","ftrojan","ina","lf","trojan","wanet","refool","badnet_sq","blend","sig"]:
            trainer = BackdoorModelTrainer(
                self.teacher_gen,
                self.args,
                
        )
            
            clean_metrics, \
            clean_test_epoch_predict_list, \
            clean_test_epoch_label_list, \
             = trainer.test_given_dataloader(self.test_loader, verbose=1)

            teacher_test_acc = clean_metrics["test_acc"]
            bd_metrics, \
            bd_test_epoch_predict_list, \
            bd_test_epoch_label_list, \
            bd_test_epoch_original_index_list, \
            bd_test_epoch_poison_indicator_list, \
            bd_test_epoch_original_targets_list = trainer.test_given_dataloader_on_mix(self.test_poi_loader, verbose=1)
            from utils.trainer_cls import all_acc
            teacher_test_poi_acc = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
            
        else:
            teacher_test_acc = self.test(
                self.teacher_gen, self.test_loader)
            teacher_test_poi_acc = self.test(
                self.teacher_gen, self.test_poi_loader)
        if self.args.shuf_teacher > 0.:
            log_file_path = self.home_path+self.args.trigger_pattern+"_"+str(self.args.seed)+"_defense.txt"
        else:
            log_file_path = self.home_path+self.args.trigger_pattern+"_"+str(self.args.seed)+".txt"
        print(f'Teacher Acc: {teacher_test_acc*100:.2f}%')
        print(f'Teacher ASR: {teacher_test_poi_acc*100:.2f}%')
        with open(log_file_path, 'w') as f:
            f.write(f'Eval/teacher_test_acc:{teacher_test_acc*100:.2f}%')
            f.write(f'Teacher ASR: {teacher_test_poi_acc*100:.2f}%')

        running_data_time, running_batch_time = AggregateScalar(), AggregateScalar()
        running_student_maxes_avg, running_teacher_maxes_avg = AggregateScalar(), AggregateScalar()
        running_student_total_loss, running_generator_total_loss = AggregateScalar(), AggregateScalar()
        student_maxes_distribution, student_argmaxes_distribution = [], []
        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
        shuffle_kl_loss_avg, shuffle_mask_avg, shuffle_drop_avg = AggregateScalar(), AggregateScalar(), AggregateScalar()


        
        # target_class={}
        # target_class["badnet_grid"] = [0]
        # target_class["badnet_sq"] = [7,0]
        # target_class["blend"] = [4,0]
        # target_class["ctrl"] = [6,0]
        # target_class["ftrojan"] = [0]
        # target_class["ina"] = [0]
        # target_class["lf"] = [0]
        # target_class["refool"] = [2,6,0]
        # target_class["sig"] = [0]
        # target_class["trojan"] = [0]
        # target_class["wanet"] = [0]
        
        
        
 

        # end = time.time()
        # idx_pseudo = 0
        # cos = nn.CosineSimilarity(dim=1)
        # self.n_pseudo_batches = 0
        # dataset_path = "/dataset/ztb/"
        # acc_save_path = dataset_path+self.args.trigger_pattern+"/"+'ori_acc.npy'
        # asr_save_path = dataset_path+self.args.trigger_pattern+"/"+'ori_asr.npy'
        # ori_acc = np.load(acc_save_path)
        # ori_asr = np.load(asr_save_path)
        

        

        # while self.n_pseudo_batches < self.args.total_n_pseudo_batches:
            
        #     x_pseudo, expected = self.generator.__next__(torch.randint(0, 10, (self.args.batch_size,)).to(self.args.device))
        #     ## Take n_generator_iter steps on generator
        #     if idx_pseudo % self.n_repeat_batch < self.args.n_generator_iter:
        #         self.student_gen.train()
        #         student_logits, *student_activations = self.student_gen(x_pseudo)
        #         teacher_logits, *teacher_activations = self.teacher_gen(x_pseudo)
        #         generator_total_loss = self.KT_loss_generator(student_logits, teacher_logits) + 0.02*F.cross_entropy(teacher_logits, F.one_hot(expected, num_classes=10).float())
                

        #         if self.shuf_loss_fn is not None:
        #             shufl_kl_loss = self.shuf_loss_fn.loss(teacher_logits, x_pseudo)
        #             generator_total_loss = generator_total_loss + shufl_kl_loss
        #             shuffle_kl_loss_avg.update(shufl_kl_loss.item(
        #             ) if torch.is_tensor(shufl_kl_loss) else shufl_kl_loss)
        #             shuffle_mask_avg.update(self.shuf_loss_fn.n_shufl_penalized)

        #         self.optimizer_generator.zero_grad()
        #         generator_total_loss.backward()
        #         torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
        #         self.optimizer_generator.step()


        #     ## Take n_student_iter steps on student
        #     elif idx_pseudo % self.n_repeat_batch < (self.args.n_generator_iter + self.args.n_student_iter):
        #         self.student_gen.train()
                
        #         with torch.no_grad():
        #             teacher_logits, *teacher_activations = self.teacher(x_pseudo)
                    
        #         if self.args.sup_backdoor and self.args.use_hg and self.n_pseudo_batches >= self.args.hg_start_round:
        #             # generate noise
        #             batch_pert = torch.zeros_like(
        #                 x_pseudo[:1], requires_grad=True, device=self.args.device)
        #             pert_opt = torch.optim.Adam(params=[batch_pert], lr=self.args.pert_lr)
        #             if self.args.sup_pert_model == 'student':
        #                 pert_model = self.student_gen
        #             elif self.args.sup_pert_model == 'teacher':
        #                 pert_model = self.teacher_gen
        #             else:
        #                 raise RuntimeError(f"self.args.sup_pert_model: {self.args.sup_pert_model}")
                    
        #             self.student_gen.train()

        #             for i_pert in range(self.args.inner_round):
        #                 sud_logits, *_ = pert_model(x_pseudo)
        #                 per_logits, *_ = pert_model(x_pseudo+batch_pert)

        #                 loss_regu = self.KT_loss_generator(per_logits, sud_logits.detach()) +0.001*torch.pow(torch.norm(batch_pert),2)
        #                 pert_opt.zero_grad()
        #                 loss_regu.backward(retain_graph=True)
        #                 pert_opt.step()
        #             pert = batch_pert  # TODO may not be useful: * min(1, 10 / torch.norm(batch_pert))

        #             # hg losses
        #             #define the inner loss L2
        #             def loss_inner(perturb, model_params):
        #                 images = x_pseudo.cuda()
        #                 per_img = images+perturb[0]
                        
        #                 with torch.no_grad():
        #                     sud_logits, *_ = self.student_gen(images)
        #                     sud_logits = sud_logits.detach()
        #                 per_logits, *_ = self.student_gen(per_img)
                        
        #                 # loss = torch.mean(cos(sud_logits, per_logits))
        #                 loss = self.KT_loss_generator(per_logits, sud_logits)
        #                 loss_regu = loss +0.001*torch.pow(torch.norm(perturb[0]),2)
        #                 return loss_regu

        #             def loss_outer(perturb, model_params):
        #                 portion = self.args.hg_out_portion # 0.02  # TODO increase this. Increase batch size.
        #                 images = x_pseudo
        #                 patching = torch.zeros_like(images, device='cuda')
        #                 number = images.shape[0]
        #                 rand_idx = random.sample(list(np.arange(number)),int(number*portion))
        #                 patching[rand_idx] = perturb[0]
        #                 unlearn_imgs = images+patching

        #                 # self.student.eval()
        #                 student_logits, *student_activations = self.student_gen(x_pseudo)
        #                 # self.student.train()
        #                 student_per_logits, *_ = self.student_gen(unlearn_imgs)

        #                 pert_loss = self.KT_loss_generator(student_per_logits, student_logits.detach())
        #                 loss = - pert_loss
                        
        #                 return loss

        #             inner_opt = hg.GradientDescent(loss_inner, 0.1)
                    
        #             # optimize
        #             self.optimizer_student_gen.zero_grad()
        #             student_params = list(self.student_gen.parameters())
        #             hg.fixed_point(pert, student_params, 5, inner_opt, loss_outer) 

        #             # bn_freezer.restore(self.student)

        #             # FIXME ad-hoc do loss separately.
        #             # self.student.train()
        #             student_logits, *student_activations = self.student_gen(x_pseudo)
        #             kl_loss = self.KT_loss_student(
        #                 student_logits, student_activations, teacher_logits, teacher_activations)+0.1*F.cross_entropy(student_logits, F.one_hot(expected, num_classes=10).float())
                    
        #             kl_loss.backward()

        #             self.optimizer_student_gen.step()

        #             with torch.no_grad():
        #                 student_total_loss = loss_outer(pert, student_params)
        #         else:
                    
        #             student_logits, *student_activations = self.student_gen(x_pseudo)
        #             student_total_loss = self.KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations)
                    
        #             # Backdoor suppression
        #             if self.args.sup_backdoor and not self.args.use_hg and self.n_pseudo_batches >= self.args.hg_start_round:
        #                 batch_pert = torch.zeros_like(
        #                     x_pseudo[:1], requires_grad=True, device='cuda')
        #                 pert_opt = torch.optim.SGD(params=[batch_pert], lr=10)

        #                 for _ in range(self.args.inner_round):
        #                     sud_logits, *_ = self.student_gen(x_pseudo)
        #                     per_logits, *_ = self.student_gen(x_pseudo+batch_pert)

        #                     loss_regu = self.KT_loss_generator(per_logits, sud_logits.detach())
        #                     pert_opt.zero_grad()
        #                     loss_regu.backward(retain_graph=True)
        #                     pert_opt.step()
        #                 pert = batch_pert * min(1, 10 / torch.norm(batch_pert))

        #                 student_per_logits, *_ = self.student_gen(x_pseudo + pert.detach())

        #                 student_total_loss = student_total_loss \
        #                     - 0.5 * self.KT_loss_generator(student_per_logits, student_logits)

        #             self.optimizer_student_gen.zero_grad()
        #             student_total_loss.backward()
        #             torch.nn.utils.clip_grad_norm_(self.student_gen.parameters(), 5)
        #             self.optimizer_student_gen.step()

        #     if self.shuf_loss_fn is not None and self.args.pseudo_test_batches > 0:
        #         # to check if the shuffle model is good enough.
        #         if self.n_pseudo_batches >= (self.args.n_generator_iter + self.args.n_student_iter) and self.n_pseudo_batches < self.args.pseudo_test_batches:
        #             self.shuf_loss_fn.add_syn_batch(x_pseudo, teacher_logits)

        #         if self.n_pseudo_batches + 1 > self.args.pseudo_test_batches:
        #             self.shuf_loss_fn.select_shuffle(self.args.shuf_teacher, 
        #                 self.test_loader, self.test_poi_loader)

        #     ## Last call to this batch, log metrics
        #     if (idx_pseudo + 1) % self.n_repeat_batch == 0:
        #         with torch.no_grad():
        #             teacher_maxes, teacher_argmaxes = torch.max(torch.softmax(teacher_logits, dim=1), dim=1)
        #             student_maxes, student_argmaxes = torch.max(torch.softmax(student_logits, dim=1), dim=1)
        #             running_generator_total_loss.update(float(generator_total_loss))
        #             running_student_total_loss.update(float(student_total_loss))
        #             running_teacher_maxes_avg.update(float(torch.mean(teacher_maxes)))
        #             running_student_maxes_avg.update(float(torch.mean(student_maxes)))
        #             teacher_maxes_distribution.append(teacher_maxes)
        #             teacher_argmaxes_distribution.append(teacher_argmaxes)
        #             student_maxes_distribution.append(student_maxes)
        #             student_argmaxes_distribution.append(student_argmaxes)

        #         if (self.n_pseudo_batches+1) % self.args.log_freq == 0 or self.n_pseudo_batches == self.args.total_n_pseudo_batches - 1:
        #             with torch.no_grad(): 
        #                 if self.args.trigger_pattern in ["bpp","ctrl","ssba","ftrojan","ina","lf","trojan","wanet","refool","badnet_sq","blend","sig"]:
        #                     trainer = BackdoorModelTrainer(
        #                         self.student_gen,
        #                         self.args,
                                
        #                 )
                            
        #                     clean_metrics, \
        #                     clean_test_epoch_predict_list, \
        #                     clean_test_epoch_label_list, \
        #                     = trainer.test_given_dataloader(self.test_loader, verbose=1)

        #                     test_acc = clean_metrics["test_acc"]
        #                     bd_metrics, \
        #                     bd_test_epoch_predict_list, \
        #                     bd_test_epoch_label_list, \
        #                     bd_test_epoch_original_index_list, \
        #                     bd_test_epoch_poison_indicator_list, \
        #                     bd_test_epoch_original_targets_list = trainer.test_given_dataloader_on_mix(self.test_poi_loader, verbose=1)
        #                     from utils.trainer_cls import all_acc
        #                     test_poi_acc = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
        #                 else:    
        #                     test_acc = self.test(self.student_gen, self.test_loader)
        #                     test_poi_acc = self.test(self.student_gen, self.test_poi_loader)
        #                 log_info = {}
        #                 print('\nBatch {}/{} -- Generator Loss: {:02.2f} -- Student Loss: {:02.2f}'.format(self.n_pseudo_batches, self.args.total_n_pseudo_batches, running_generator_total_loss.avg(), running_student_total_loss.avg()))
        #                 print_info = 'Test Acc: {:02.2f}%'.format(test_acc*100)

                        
        #                 print_info += f' ASR: {test_poi_acc*100:.2f}%'
        #                 log_info.update({'Eval/test_poi_acc': test_poi_acc*100, })
                        
        #                 print_info += f" stu lr: {self.scheduler_student_gen.get_last_lr()[0]:g}, " \
        #                     f"gen lr: {self.scheduler_generator.get_last_lr()[0]:g}"
        #                 print(print_info)
                        
        #                 log_info.update({
        #                     'TRAIN_PSEUDO/generator_total_loss': running_generator_total_loss.avg(),
        #                     'TRAIN_PSEUDO/student_total_loss': running_student_total_loss.avg(),
        #                     'TRAIN_PSEUDO/teacher_maxes_avg': running_teacher_maxes_avg.avg(),
        #                     'TRAIN_PSEUDO/student_maxes_avg': running_student_maxes_avg.avg(),
        #                     'TRAIN_PSEUDO/student_lr': self.scheduler_student_gen.get_last_lr()[0],
        #                     'TRAIN_PSEUDO/generator_lr': self.scheduler_generator.get_last_lr()[0],
        #                     'TIME/data_time_sec': running_data_time.avg(),
        #                     'TIME/batch_time_sec': running_batch_time.avg(),
        #                     'Eval/test_acc': test_acc*100,
        #                     'TRAIN_SHUFFLE/shuffle_kl_loss': shuffle_kl_loss_avg.avg(),
        #                     'TRAIN_SHUFFLE/shuffle_kl_loss_std': shuffle_kl_loss_avg.std(),
        #                     'TRAIN_SHUFFLE/shuffle_mask': shuffle_mask_avg.avg(),
        #                     'TRAIN_SHUFFLE/shuffle_mask_std': shuffle_mask_avg.std(),
        #                     'TRAIN_SHUFFLE/shuffle_drop': shuffle_drop_avg.avg(),
        #                 })
                        
        #                 # epochs.append(self.n_pseudo_batches)
        #                 # our_acc.append(test_acc)
        #                 # our_asr.append(test_poi_acc)
        #                 with open(log_file_path, 'a') as f:
        #                     f.write('\nBatch {}/{} -- Generator Loss: {:02.2f} -- Student Loss: {:02.2f}'.format(self.n_pseudo_batches, self.args.total_n_pseudo_batches, running_generator_total_loss.avg(), running_student_total_loss.avg()))
        #                     f.write('Test Acc: {:02.2f}%'.format(test_acc*100))
        #                     f.write(f' ASR: {test_poi_acc*100:.2f}%')
        #                     for key, value in log_info.items():
        #                         f.write(f'{key}: {value}\n')


        #                 running_data_time.reset(), running_batch_time.reset()
        #                 running_teacher_maxes_avg.reset(), running_student_maxes_avg.reset()
        #                 running_generator_total_loss.reset(), running_student_total_loss.reset(),
        #                 teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
        #                 student_maxes_distribution, student_argmaxes_distribution = [], []
        #                 shuffle_kl_loss_avg.reset(), shuffle_mask_avg.reset(), shuffle_drop_avg.reset()
                    

        #             if self.args.save_n_checkpoints > 1:
        #                 if (self.n_pseudo_batches+1) // self.args.log_freq > self.args.log_times - self.args.save_n_checkpoints:
        #                     self.save_model(log_info, f"pb{self.n_pseudo_batches}.pth")
        #             self.save_model(log_info, f"last.pth")

        #         self.n_pseudo_batches += 1
        #         self.scheduler_student_gen.step()
        #         self.scheduler_generator.step()

        #     idx_pseudo += 1
        #     running_batch_time.update(time.time() - end)
        #     end = time.time()
        # self.save_generator_new()
        

        # fname = "/home/ztb/ABD/zskt/generator/"+self.args.trigger_pattern+"_"+self.args.teacher_architecture+"_"+self.args.student_architecture+"_generator_new.pth"
        # self.generator.load_state_dict(torch.load(fname)['state_dict'])
        fname = "/home/ztb/ABD/zskt/generator/"+self.args.trigger_pattern+"_generator_new.pth"
            
        self.generator.load_state_dict(torch.load(fname)['state_dict'])
        #after sample set created goto the zskt process
        running_data_time, running_batch_time = AggregateScalar(), AggregateScalar()
        running_student_maxes_avg, running_teacher_maxes_avg = AggregateScalar(), AggregateScalar()
        running_student_total_loss, running_generator_total_loss = AggregateScalar(), AggregateScalar()
        student_maxes_distribution, student_argmaxes_distribution = [], []
        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
        shuffle_kl_loss_avg, shuffle_mask_avg, shuffle_drop_avg = AggregateScalar(), AggregateScalar(), AggregateScalar()

        end = time.time()
        idx_pseudo = 0
        cos = nn.CosineSimilarity(dim=1)
        self.n_pseudo_batches = 0
        dataset_path = "/dataset/ztb/"

        
        ori_shape = (128,3,32,32)
       
        # 保存路径
        dataset_path = "/dataset/ztb/"
        acc_save_path = dataset_path+self.args.trigger_pattern+"/"+'ori_acc.npy'
        asr_save_path = dataset_path+self.args.trigger_pattern+"/"+'ori_asr.npy'
        ori_acc = np.load(acc_save_path)
        ori_asr = np.load(asr_save_path)
        ext_value = 0.925
        sample_set_size = 40000    
        generated_dataset = GeneratedDataset(self.args, sample_set_size,ori_shape)
        print("training dataset generated!")
        
        target_class={}
        target_class["badnet_grid"] = [0]
        target_class["badnet_sq"] = [7,0]
        target_class["blend"] = [4,0]
        target_class["ctrl"] = [6,0]
        target_class["ftrojan"] = [0]
        target_class["ina"] = [0]
        target_class["lf"] = [0]
        target_class["refool"] = [2,6,0]
        target_class["sig"] = [0]
        target_class["trojan"] = [0]
        target_class["wanet"] = [0]
        
        
        ext_lim = 0
        result = torch.tensor([]).to(self.args.device)
        for idx in range(0,int(sample_set_size*0.1)):
        #for idx in range(5000,5000+200):
            if idx%1000 ==0 :
                print('\n processed {} pictures/total {} pic... '.format(idx, int(sample_set_size*0.1)))
            #input = generated_dataset.__getitem__(idx).to(self.args.device)
            input, expected = self.generator.__next__(torch.randint(0, 10, (self.args.batch_size,)).to(self.args.device))
            teacher_logits, *teacher_activations = self.teacher(input)
            for logits in teacher_logits.data:
                result = torch.cat((result, logits),dim=0)
        ext_lim = torch.quantile(result, ext_value)
        target_class_list = target_class[self.args.trigger_pattern]
        print(target_class_list)
                    
        
        #ext_lim = torch.quantile(result, 0.88)
            
        supplement = {i: [] for i in target_class_list}
        for idx in range(5000):
            x_pseudo, expected = self.generator.__next__(torch.randint(0, 1, (self.args.batch_size,)).to(self.args.device))
            teacher_logits, *teacher_activations = self.teacher(x_pseudo)
            for i in range(teacher_logits.size(0)):
                # 找到最大值的索引
                max_index = torch.argmax(teacher_logits[i])
                if max_index.item() in target_class_list and teacher_logits[i][max_index] < ext_lim:
                    supplement[max_index.item()].append(x_pseudo[i].detach().cpu().numpy())
        i=0
        print("now have supplement set with size:")
        for target in target_class_list:
            print(len(supplement[target]))
        while self.n_pseudo_batches < self.args.total_n_pseudo_batches:
            
            x_pseudo, expected = self.generator.__next__(torch.randint(0, 10, (self.args.batch_size,)).to(self.args.device))
            ## Take n_generator_iter steps on generator
            # if idx_pseudo % self.n_repeat_batch < self.args.n_generator_iter:
            #     self.student_gen.train()
            #     student_logits, *student_activations = self.student_gen(x_pseudo)
            #     teacher_logits, *teacher_activations = self.teacher_gen(x_pseudo)
            #     generator_total_loss = self.KT_loss_generator(student_logits, teacher_logits) + 0.02*F.cross_entropy(teacher_logits, F.one_hot(expected, num_classes=10).float())
                

            #     if self.shuf_loss_fn is not None:
            #         shufl_kl_loss = self.shuf_loss_fn.loss(teacher_logits, x_pseudo)
            #         generator_total_loss = generator_total_loss + shufl_kl_loss
            #         shuffle_kl_loss_avg.update(shufl_kl_loss.item(
            #         ) if torch.is_tensor(shufl_kl_loss) else shufl_kl_loss)
            #         shuffle_mask_avg.update(self.shuf_loss_fn.n_shufl_penalized)

            #     self.optimizer_generator.zero_grad()
            #     generator_total_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
            #     self.optimizer_generator.step()


            ## Take n_student_iter steps on student
            if idx_pseudo % self.n_repeat_batch < (self.args.n_generator_iter + self.args.n_student_iter):
                self.student_gen.train()
                # if self.n_pseudo_batches % 5 ==0:
                #                 supplement={i: [] for i in target_class_list}
                with torch.no_grad():
                    teacher_logits, *teacher_activations = self.teacher(x_pseudo)
                    x_pseudo_np = x_pseudo.detach().cpu().numpy()
                    for k in range(teacher_logits.size(0)):
                   # 找到最大值的索引
                        max_index = torch.argmax(teacher_logits[k])
                        if max_index.item() in target_class_list and teacher_logits[k][max_index] > ext_lim:
                        #     #print("found ext in"+str(self.n_pseudo_batches)+", classified as class "+str(max_index.item()))
                        #     #print(str(teacher_logits[k][max_index]))
                        #     while len(supplement[max_index.item()])==0:
                        #         x_pseudo_new , expected_new = self.generator.__next__(torch.full((self.args.batch_size,), max_index.item()).to(self.args.device))
                        #         teacher_logits_new, *teacher_activations_new = self.teacher(x_pseudo_new)
                        #         count= 0
                        #         for j in range(teacher_logits_new.size(0)):
                        #             max_index_new = torch.argmax(teacher_logits_new[j])
                        #             #print("classified as class:"+str(max_index_new.item()))
                        #             if max_index_new.item() == max_index.item() and teacher_logits_new[j][max_index_new] <=ext_lim:
                        #                 #print("the one classified as target class have logit as:"+str(teacher_logits_new[j][max_index_new]))
                        #                 count+=1
                        #                 supplement[max_index.item()].append(x_pseudo_new[j].detach().cpu().numpy())
                                #print(str(count)+"/128 are classified as class" + str(max_index.item()))
                        #print("found supplement in"+str(test)+" ,class: "+ str(max_index_new))
                            # x_pseudo_np[k] = supplement[max_index.item()].pop()
                            x_pseudo_np[k] = supplement[max_index.item()][np.random.randint(0, len(supplement))]
                    teacher_logits, *teacher_activations = self.teacher(torch.from_numpy(x_pseudo_np).to(self.args.device))
                    # teacher_logits, *teacher_activations = self.teacher(x_pseudo)
                    
                #     # 获取最大值及其索引
                #     max_values, max_indices = torch.max(teacher_logits, dim=1, keepdim=True)

                #     # 检查最大值的索引是否在target_class_list中
                #     max_in_target_class = torch.isin(max_indices.squeeze(), torch.tensor(target_class_list).to(self.args.device))


                #     # 创建一个布尔掩码，指示最大值不在target_class_list中的行
                #     max_not_in_target_class = ~max_in_target_class
                    
                #     T = 2.0

                #     # 对teacher_logits应用softmax并缩放，保留每行的排序信息，但缩小差异
                #     ori = teacher_logits[max_not_in_target_class, :]
                #    # 计算每一行的均值 
                #     sum_values = teacher_logits.sum(dim=1, keepdim=True) - max_values
                #     mean_values = sum_values / (teacher_logits.size(1) - 1)
                    

                #     smoothed_logits = (teacher_logits[max_not_in_target_class, :]-mean_values[max_not_in_target_class]) / T
                    
                #     ori_teacher_activations = teacher_activations
                #     teacher_logits[max_not_in_target_class, :] = smoothed_logits + mean_values[max_not_in_target_class]
                max_values, max_indices = torch.max(teacher_logits, dim=1, keepdim=True)

                # Create a mask that identifies the positions of the maximum values
                mask = torch.zeros_like(teacher_logits, dtype=torch.bool, device=self.args.device)
                mask.scatter_(1, max_indices, True)

                # Create a mask for the positions in target_class_list
                target_class_mask = torch.zeros_like(teacher_logits, dtype=torch.bool, device=self.args.device)
                for target_class in target_class_list:
                    target_class_mask[:, target_class] = True

                # Create a mask for values to exclude (target_class_list and max values)
                exclude_mask = torch.ones_like(teacher_logits, dtype=torch.bool, device=self.args.device)

                # Exclude max values and target_class_list positions
                for target_class in target_class_list:
                    exclude_mask[:, target_class] = False
                exclude_mask.scatter_(1, max_indices, False)

                # Calculate the sum of each row excluding the values in target_class_list and max values
                sum_values = teacher_logits * exclude_mask  # Masking out excluded positions
                sum_values = sum_values.sum(dim=1, keepdim=True)

                # Calculate how many valid values are left for each row (after excluding max value and target_class_list positions)
                num_excluded = exclude_mask.sum(dim=1, keepdim=True) - 1  # subtract 1 for the max value

                # Avoid division by zero if all values are excluded
                mean_values = sum_values / torch.clamp(num_excluded, min=1)

                # Create a mask for when the max value is not in target_class_list
                max_not_in_target = ~target_class_mask.gather(1, max_indices)

                # Replace the target_class_list values with the calculated mean where max value is not in the target positions
                teacher_logits = torch.where(max_not_in_target & target_class_mask, mean_values, teacher_logits)
                if self.args.sup_backdoor and self.args.use_hg and self.n_pseudo_batches >= self.args.hg_start_round:
                    # generate noise
                    batch_pert = torch.zeros_like(
                        x_pseudo[:1], requires_grad=True, device=self.args.device)
                    pert_opt = torch.optim.Adam(params=[batch_pert], lr=self.args.pert_lr)
                    if self.args.sup_pert_model == 'student':
                        pert_model = self.student_gen
                    elif self.args.sup_pert_model == 'teacher':
                        pert_model = self.teacher_gen
                    else:
                        raise RuntimeError(f"self.args.sup_pert_model: {self.args.sup_pert_model}")
                    
                    self.student_gen.train()

                    for i_pert in range(self.args.inner_round):
                        sud_logits, *_ = pert_model(x_pseudo)
                        per_logits, *_ = pert_model(x_pseudo+batch_pert)

                        loss_regu = self.KT_loss_generator(per_logits, sud_logits.detach()) +0.001*torch.pow(torch.norm(batch_pert),2)
                        pert_opt.zero_grad()
                        loss_regu.backward(retain_graph=True)
                        pert_opt.step()
                    pert = batch_pert  # TODO may not be useful: * min(1, 10 / torch.norm(batch_pert))

                    # hg losses
                    #define the inner loss L2
                    def loss_inner(perturb, model_params):
                        images = x_pseudo.cuda()
                        per_img = images+perturb[0]
                        
                        with torch.no_grad():
                            sud_logits, *_ = self.student_gen(images)
                            sud_logits = sud_logits.detach()
                        per_logits, *_ = self.student_gen(per_img)
                        
                        # loss = torch.mean(cos(sud_logits, per_logits))
                        loss = self.KT_loss_generator(per_logits, sud_logits)
                        loss_regu = loss +0.001*torch.pow(torch.norm(perturb[0]),2)
                        return loss_regu

                    def loss_outer(perturb, model_params):
                        portion = self.args.hg_out_portion # 0.02  # TODO increase this. Increase batch size.
                        images = x_pseudo
                        patching = torch.zeros_like(images, device='cuda')
                        number = images.shape[0]
                        rand_idx = random.sample(list(np.arange(number)),int(number*portion))
                        patching[rand_idx] = perturb[0]
                        unlearn_imgs = images+patching

                        # self.student.eval()
                        student_logits, *student_activations = self.student_gen(x_pseudo)
                        # self.student.train()
                        student_per_logits, *_ = self.student_gen(unlearn_imgs)

                        pert_loss = self.KT_loss_generator(student_per_logits, student_logits.detach())
                        loss = - pert_loss
                        
                        return loss

                    inner_opt = hg.GradientDescent(loss_inner, 0.1)
                    
                    # optimize
                    self.optimizer_student_gen.zero_grad()
                    student_params = list(self.student_gen.parameters())
                    hg.fixed_point(pert, student_params, 5, inner_opt, loss_outer) 

                    # bn_freezer.restore(self.student)

                    # FIXME ad-hoc do loss separately.
                    # self.student.train()
                    student_logits, *student_activations = self.student_gen(x_pseudo)
                    kl_loss = self.KT_loss_student(
                        student_logits, student_activations, teacher_logits, teacher_activations)+0.1*F.cross_entropy(student_logits, F.one_hot(expected, num_classes=10).float())
                    
                    kl_loss.backward()

                    self.optimizer_student_gen.step()

                    with torch.no_grad():
                        student_total_loss = loss_outer(pert, student_params)
                else:
                    
                    student_logits, *student_activations = self.student_gen(x_pseudo)
                    student_total_loss = self.KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations)
                    
                    # Backdoor suppression
                    if self.args.sup_backdoor and not self.args.use_hg and self.n_pseudo_batches >= self.args.hg_start_round:
                        batch_pert = torch.zeros_like(
                            x_pseudo[:1], requires_grad=True, device='cuda')
                        pert_opt = torch.optim.SGD(params=[batch_pert], lr=10)

                        for _ in range(self.args.inner_round):
                            sud_logits, *_ = self.student_gen(x_pseudo)
                            per_logits, *_ = self.student_gen(x_pseudo+batch_pert)

                            loss_regu = self.KT_loss_generator(per_logits, sud_logits.detach())
                            pert_opt.zero_grad()
                            loss_regu.backward(retain_graph=True)
                            pert_opt.step()
                        pert = batch_pert * min(1, 10 / torch.norm(batch_pert))

                        student_per_logits, *_ = self.student_gen(x_pseudo + pert.detach())

                        student_total_loss = student_total_loss \
                            - 0.5 * self.KT_loss_generator(student_per_logits, student_logits)

                    self.optimizer_student_gen.zero_grad()
                    student_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_gen.parameters(), 5)
                    self.optimizer_student_gen.step()

            if self.shuf_loss_fn is not None and self.args.pseudo_test_batches > 0:
                # to check if the shuffle model is good enough.
                if self.n_pseudo_batches >= (self.args.n_generator_iter + self.args.n_student_iter) and self.n_pseudo_batches < self.args.pseudo_test_batches:
                    self.shuf_loss_fn.add_syn_batch(x_pseudo, teacher_logits)

                if self.n_pseudo_batches + 1 > self.args.pseudo_test_batches:
                    self.shuf_loss_fn.select_shuffle(self.args.shuf_teacher, 
                        self.test_loader, self.test_poi_loader)

            ## Last call to this batch, log metrics
            if (idx_pseudo + 1) % self.n_repeat_batch == 0:
                with torch.no_grad():
                    teacher_maxes, teacher_argmaxes = torch.max(torch.softmax(teacher_logits, dim=1), dim=1)
                    student_maxes, student_argmaxes = torch.max(torch.softmax(student_logits, dim=1), dim=1)
                    #running_generator_total_loss.update(float(generator_total_loss))
                    running_student_total_loss.update(float(student_total_loss))
                    running_teacher_maxes_avg.update(float(torch.mean(teacher_maxes)))
                    running_student_maxes_avg.update(float(torch.mean(student_maxes)))
                    teacher_maxes_distribution.append(teacher_maxes)
                    teacher_argmaxes_distribution.append(teacher_argmaxes)
                    student_maxes_distribution.append(student_maxes)
                    student_argmaxes_distribution.append(student_argmaxes)

                if (self.n_pseudo_batches+1) % self.args.log_freq == 0 or self.n_pseudo_batches == self.args.total_n_pseudo_batches - 1:
                    with torch.no_grad(): 
                        if self.args.trigger_pattern in ["bpp","ctrl","ssba","ftrojan","ina","lf","trojan","wanet","refool","badnet_sq","blend","sig"]:
                            trainer = BackdoorModelTrainer(
                                self.student_gen,
                                self.args,
                                
                        )
                            
                            clean_metrics, \
                            clean_test_epoch_predict_list, \
                            clean_test_epoch_label_list, \
                            = trainer.test_given_dataloader(self.test_loader, verbose=1)

                            test_acc = clean_metrics["test_acc"]
                            bd_metrics, \
                            bd_test_epoch_predict_list, \
                            bd_test_epoch_label_list, \
                            bd_test_epoch_original_index_list, \
                            bd_test_epoch_poison_indicator_list, \
                            bd_test_epoch_original_targets_list = trainer.test_given_dataloader_on_mix(self.test_poi_loader, verbose=1)
                            from utils.trainer_cls import all_acc
                            test_poi_acc = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
                        else:    
                            test_acc = self.test(self.student_gen, self.test_loader)
                            test_poi_acc = self.test(self.student_gen, self.test_poi_loader)
                        log_info = {}
                        print('\nBatch {}/{} -- Generator Loss: {:02.2f} -- Student Loss: {:02.2f}'.format(self.n_pseudo_batches, self.args.total_n_pseudo_batches, running_generator_total_loss.avg(), running_student_total_loss.avg()))
                        print_info = 'Test Acc: {:02.2f}%'.format(test_acc*100)

                        
                        print_info += f' ASR: {test_poi_acc*100:.2f}%'
                        log_info.update({'Eval/test_poi_acc': test_poi_acc*100, })
                        
                        print_info += f" stu lr: {self.scheduler_student_gen.get_last_lr()[0]:g}, " \
                            f"gen lr: {self.scheduler_generator.get_last_lr()[0]:g}"
                        print(print_info)
                        
                        log_info.update({
                            'TRAIN_PSEUDO/generator_total_loss': running_generator_total_loss.avg(),
                            'TRAIN_PSEUDO/student_total_loss': running_student_total_loss.avg(),
                            'TRAIN_PSEUDO/teacher_maxes_avg': running_teacher_maxes_avg.avg(),
                            'TRAIN_PSEUDO/student_maxes_avg': running_student_maxes_avg.avg(),
                            'TRAIN_PSEUDO/student_lr': self.scheduler_student_gen.get_last_lr()[0],
                            'TRAIN_PSEUDO/generator_lr': self.scheduler_generator.get_last_lr()[0],
                            'TIME/data_time_sec': running_data_time.avg(),
                            'TIME/batch_time_sec': running_batch_time.avg(),
                            'Eval/test_acc': test_acc*100,
                            'TRAIN_SHUFFLE/shuffle_kl_loss': shuffle_kl_loss_avg.avg(),
                            'TRAIN_SHUFFLE/shuffle_kl_loss_std': shuffle_kl_loss_avg.std(),
                            'TRAIN_SHUFFLE/shuffle_mask': shuffle_mask_avg.avg(),
                            'TRAIN_SHUFFLE/shuffle_mask_std': shuffle_mask_avg.std(),
                            'TRAIN_SHUFFLE/shuffle_drop': shuffle_drop_avg.avg(),
                        })
                        
                        epochs.append(self.n_pseudo_batches)
                        our_acc.append(test_acc)
                        our_asr.append(test_poi_acc)
                        with open(log_file_path, 'a') as f:
                            f.write('\nBatch {}/{} -- Generator Loss: {:02.2f} -- Student Loss: {:02.2f}'.format(self.n_pseudo_batches, self.args.total_n_pseudo_batches, running_generator_total_loss.avg(), running_student_total_loss.avg()))
                            f.write('Test Acc: {:02.2f}%'.format(test_acc*100))
                            f.write(f' ASR: {test_poi_acc*100:.2f}%')
                            for key, value in log_info.items():
                                f.write(f'{key}: {value}\n')


                        running_data_time.reset(), running_batch_time.reset()
                        running_teacher_maxes_avg.reset(), running_student_maxes_avg.reset()
                        running_generator_total_loss.reset(), running_student_total_loss.reset(),
                        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
                        student_maxes_distribution, student_argmaxes_distribution = [], []
                        shuffle_kl_loss_avg.reset(), shuffle_mask_avg.reset(), shuffle_drop_avg.reset()
                    

                    if self.args.save_n_checkpoints > 1:
                        if (self.n_pseudo_batches+1) // self.args.log_freq > self.args.log_times - self.args.save_n_checkpoints:
                            self.save_model(log_info, f"pb{self.n_pseudo_batches}.pth")
                    self.save_model(log_info, f"last.pth")

                self.n_pseudo_batches += 1
                self.scheduler_student_gen.step()
                self.scheduler_generator.step()

            idx_pseudo += 1
            running_batch_time.update(time.time() - end)
            end = time.time()
        

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))

        # 绘制 ori_acc
        plt.plot(epochs, ori_acc, marker='o', linestyle='-', label='Ori Acc', color='blue')
        # 绘制 ori_asr
        plt.plot(epochs, ori_asr, marker='o', linestyle='-', label='Ori ASR', color='red')
        # 绘制 our_acc
        plt.plot(epochs, our_acc, marker='x', linestyle='-', label='Our Acc newGen', color='green')
        # 绘制 our_asr
        plt.plot(epochs, our_asr, marker='x', linestyle='-', label='Our ASR newGen', color='orange')
        # 绘制 our_acc
        
        
        # 保存列表为 .npy 文件
        
        # 添加标题和标签
        plt.title('Accuracy and Attack Success Rate over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy / ASR')
        plt.legend()

        # 标注 teacher_test_acc 和 teacher_test_poi_acc
        plt.text(0.5, 0.6, 'Teacher Test Acc: ' + str(teacher_test_acc), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.5, 0.55, f'Teacher Test Poi Acc: '+str(teacher_test_poi_acc), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # 显示网格
        plt.grid(True)

        # 保存图表
        output_path = self.home_path+self.args.trigger_pattern+"_"+str(self.args.seed)+"_"+str(ext_value)+"_newGenwithStu_final.png" 
        plt.savefig(output_path)
        acc_save_path = dataset_path+self.args.trigger_pattern+"/"
        asr_save_path = dataset_path+self.args.trigger_pattern+"/"
        np.save(acc_save_path+'our_acc_'+str(ext_value)+'_newGenwithStu_final.npy', our_acc)
        np.save(asr_save_path+'our_asr_'+str(ext_value)+'newGenwithStu_final.npy', our_asr)
        test_acc = self.test(self.student, self.test_loader)
        if self.args.save_final_model:  # make sure last epoch saved
            self.save_model(log_info)
        print("")
        return test_acc*100


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
            self.student.load_state_dict(checkpoint['student_state_dict'])
            if not ignore_student_opt:
                self.optimizer_student.load_state_dict(checkpoint['optimizer_student'])
                self.scheduler_student.load_state_dict(checkpoint['scheduler_student'])
            if 'shuf_loss_fn' in checkpoint:
                self.shuf_loss_fn.load_state_dict(checkpoint['shuf_loss_fn'])
        else:
            raise FileNotFoundError(f"Not found checkpoint at {checkpoint_path}")

    def save_model(self, info, file_name=None):
        make_if_not_exist(self.save_model_path)
        save_dict = {'args': self.args,
                    'n_pseudo_batches': self.n_pseudo_batches,
                    'generator_state_dict': self.generator.state_dict(),
                    'student_state_dict': self.student.state_dict(),
                    'optimizer_generator': self.optimizer_generator.state_dict(),
                    'optimizer_student': self.optimizer_student.state_dict(),
                    'scheduler_generator': self.scheduler_generator.state_dict(),
                    'scheduler_student': self.scheduler_student.state_dict(),
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
        fname = "/home/ztb/ABD/zskt/generator/"+self.args.trigger_pattern+"_generator.pth"
        torch.save({'state_dict': self.generator.state_dict()}, fname)
        print(f"Save generator to {fname}")
    def save_generator_new(self):
        fname = "/home/ztb/ABD/zskt/generator/"+self.args.trigger_pattern+"_"+self.args.teacher_architecture+"_"+self.args.student_architecture+"_generator_new.pth"
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
    
    