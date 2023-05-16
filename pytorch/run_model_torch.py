import os
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchmetrics

from gbm_project.pytorch.dataset_class import DatasetGenerator
from gbm_project.pytorch.resnet_torch import ResNet3D18, ResNet3D50
from gbm_project.pytorch.resnet_spottune import resnet_spottune, resnet_agent

from MedicalNet.models import resnet
from gbm_project.pytorch.spottune_layer_translation_cfg import layer_loop, layer_loop_downsample

MODELS = ['ResNet18',
          'ResNet50']
TRANSFER_MODELS = ['MedResNet18',
                   'MedResNet50',
                   'MedResNet101',
                   'ResNet50_torch',
                   'spottune']

class RunModel(object):
    def __init__(self, config, gen_params):
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']
        self.spottune = config['spottune']
        self.n_channels = gen_params['n_channels']
        self.n_classes = config['n_classes']
        self.gumbel_temperature = config['gumbel_temperature']
        self.policy = None
        self.gen_params = gen_params
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.seed = gen_params['seed']
        self.seed_switch = config['seed_switch']
        self.seed_steps = config['seed_steps']
        self.seed_vals = config['seed_vals']
        self.temp_steps = config['temp_steps']
        self.temp_vals = config['temp_vals']

        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.acc_fn = torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(self.device)
        self.auc_fn = torchmetrics.classification.BinaryAUROC().to(self.device)

        self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f"logs are located at: {self.log_dir}")
        self.writer = SummaryWriter(self.log_dir)
        print('remember to set the data')
        self.epoch = 0
        self.best_acc =0.
        self.best_loss = 1.0
        self.best_sum = 0.


    def set_model(self, model_name='ResNet18', transfer=False):
        if not transfer:
            if model_name == 'ResNet18':
                self.model = ResNet3D18(in_channels=self.gen_params['n_channels'], 
                                        dropout=self.config['dropout']).to(self.device)
            elif model_name == 'ResNet50':
                self.model = ResNet3D50(in_channels=self.gen_params['n_channels'], 
                                        dropout=self.config['dropout']).to(self.device)
            else:
                print("no model chosen, choose from:")
                print(f"{MODELS}")
        ###########################################################################################
        #setup of models used in transfer learning. Will need to make a function later to clear up the clutter
        elif transfer:
            freeze_ignore = self.config['no_freeze']
            if model_name == 'MedResNet18':
                self.model = resnet.resnet18(sample_input_D=self.config['dim'][0],
                                             sample_input_H=self.config['dim'][1],
                                             sample_input_W=self.config['dim'][2],
                                             num_seg_classes=1,
                                             shortcut_type='B').to(self.device)
                # remove default conv_seg, replace with dense classification layers
                initial_state = torch.load('./MedicalNet/pretrain/resnet_18.pth', map_location=self.device)['state_dict']
                # remove module prefix from keys
                fixed_state = OrderedDict()
                for k, v in initial_state.items():
                    fixed_state['.'.join(k.split('.')[1:])] = v
                if self.n_channels > 1:
                    fixed_state['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)/self.n_channels
                self.model.load_state_dict(fixed_state, strict=False)
                self.freeze_layers(ignore=freeze_ignore)

#####################################################################################################
            elif model_name == 'MedResNet50':
                self.model = resnet.resnet50(sample_input_D=self.config['dim'][0],
                                             sample_input_H=self.config['dim'][1],
                                             sample_input_W=self.config['dim'][2],
                                             num_seg_classes=1,
                                             shortcut_type='B').to(self.device)
                # remove default conv_seg, replace with dense classification layers
                initial_state = torch.load('./MedicalNet/pretrain/resnet_50.pth', map_location=self.device)['state_dict']
                # remove module prefix from keys
                fixed_state = OrderedDict()
                for k, v in initial_state.items():
                    fixed_state['.'.join(k.split('.')[1:])] = v
                if self.n_channels > 1:
                    fixed_state['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)/self.n_channels
                self.model.load_state_dict(fixed_state, strict=False)
                self.freeze_layers(ignore=freeze_ignore)

######################################################################################################
            elif model_name == 'MedResNet101':
                self.model = resnet.resnet101(sample_input_D=self.config['dim'][0],
                                             sample_input_H=self.config['dim'][1],
                                             sample_input_W=self.config['dim'][2],
                                             num_seg_classes=1,
                                             shortcut_type='B').to(self.device)
                # remove default conv_seg, replace with dense classification layers
                initial_state = torch.load('./MedicalNet/pretrain/resnet_101.pth', map_location=self.device)['state_dict']
                # remove module prefix from keys
                fixed_state = OrderedDict()
                for k, v in initial_state.items():
                    fixed_state['.'.join(k.split('.')[1:])] = v
                if self.n_channels > 1:
                    fixed_state['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)/self.n_channels
                self.model.load_state_dict(fixed_state, strict=False)
                self.freeze_layers(ignore=freeze_ignore)

######################################################################################################
            elif model_name == 'ResNet50_torch':
                self.model = ResNet3D50(in_channels=self.gen_params['n_channels'], 
                    dropout=self.config['dropout']).to(self.device)

                initial_state = torchvision.models.resnet50(weights='DEFAULT').state_dict()

                fixed_state = OrderedDict()
                for name, w in initial_state.items():
                    if 'layer' in name and any(i in name for i in ('downsample', 'conv1', 'conv3')):
                        fixed_state[name] = w.unsqueeze(-1)
                    elif 'conv1' in name:
                        fixed_state[name] = w.unsqueeze(-1).transpose(-1, 2).repeat(1,1,7,1,1)/7
                    elif 'conv2' in name:
                        fixed_state[name] = w.unsqueeze(-1).transpose(-1, 2).repeat(1,1,3,1,1)/3
                    else:
                        fixed_state[name] = w

                self.model.load_state_dict(fixed_state, strict=False)
                self.freeze_layers(ignore=freeze_ignore)

######################################################################################################
            elif model_name == 'spottune':
                if not self.spottune:
                    self.spottune = True
                self.model = resnet_spottune(num_classes=self.n_classes, in_channels=self.n_channels, dropout=self.config['dropout']).to(self.device)

                initial_state = torch.load('./MedicalNet/pretrain/resnet_50.pth', map_location=self.device)['state_dict']

                fixed_state = OrderedDict()
                for k, v in initial_state.items():
                    if 'layer' in k:
                        mod_name = k.replace('module', 'blocks')
                    else:
                        mod_name = k.replace('module.', '')
                    for name, new in layer_loop.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, new)
                    for name, new in layer_loop_downsample.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, new)
                    fixed_state[mod_name] = v

                fixed_state_v2 = OrderedDict()
                for k, v in fixed_state.items():
                    fixed_state_v2[k] = v
                    fixed_state_v2[k.replace('blocks', 'parallel_blocks')] = v

                if self.n_channels > 1:
                    fixed_state_v2['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)/self.n_channels
                    #fixed_state_v2['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)

                self.model.load_state_dict(fixed_state_v2, strict=False)
                self.freeze_layers(ignore=['classify', 'parallel_blocks'])

###########################################################################################################
######################################################################################################
            elif model_name == 'spottune_imagenet':
                if not self.spottune:
                    self.spottune = True
                self.model = resnet_spottune(num_classes=self.n_classes, in_channels=self.n_channels, dropout=self.config['dropout']).to(self.device)

                initial_state = torchvision.models.resnet50(weights='DEFAULT').state_dict()

                fixed_state = OrderedDict()
                for k, v in initial_state.items():
                    mod_name = k
                    parallel_mod_name = k
                    for name, new in layer_loop.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, f"blocks.{new}")
                            parallel_mod_name = mod_name.replace(name, f"parallel_blocks.{new}")
                    for name, new in layer_loop_downsample.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, new)
                    fixed_state[mod_name] = v
                    fixed_state[parallel_mod_name] = v

                for name, w in fixed_state.items():
                    if 'blocks' in name and any(i in name for i in ('downsample', 'conv1', 'conv3')):
                        fixed_state[name] = w.unsqueeze(-1)
                    elif 'conv1' in name:
                        #fixed_state[name] = w.unsqueeze(-1).transpose(-1, 2).repeat(1,1,7,1,1)
                        fixed_state[name] = w.unsqueeze(-1).repeat(1,1,1,1,7)
                    elif 'conv2' in name:
                        #fixed_state[name] = w.unsqueeze(-1).transpose(-1, 2).repeat(1,1,3,1,1)
                        fixed_state[name] = w.unsqueeze(-1).repeat(1,1,1,1,3)
                    else:
                        fixed_state[name] = w

                self.model.load_state_dict(fixed_state, strict=False)
                self.freeze_layers(ignore=['classify', 'parallel_blocks'])

###########################################################################################################
            else:
                print("no transfer model chosen, choose from:")
                print(f"{TRANSFER_MODELS}")
             
        # set the optimizer here, since it needs the model parameters in its initialization
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config['learning_rate'], weight_decay=self.config['l2_reg'])
        #self.lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.config['lr_factor'], patience=self.config['lr_patience'], verbose=True)
        self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['lr_steps'], gamma=self.config['lr_factor'], verbose=True)


    def set_agent(self):
        self.agent = resnet_agent(num_classes=(sum(self.model.layers)*2), in_channels=self.n_channels, dropout=self.config['agent_dropout']).to(self.device)
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.config['agent_learning_rate'], weight_decay=self.config['agent_l2_reg'])
        #self.agent_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.agent_optimizer, factor=self.config['agent_lr_factor'], patience=self.config['agent_lr_patience'], verbose=True)
        self.agent_lr_sched = torch.optim.lr_scheduler.MultiStepLR(self.agent_optimizer, milestones=self.config['lr_steps'], gamma=self.config['lr_factor'], verbose=True)


    def freeze_layers(self, ignore=['conv_seg']):
        '''
        Freeze all layers except for those in the ignore list. Function can be tuned later when attempting to implement the adaptive transfer learning
        '''
        #make sure modules in the ignore list are still trainable
        for name, p in self.model.named_parameters():
            if any([i in name for i in ignore]):
                p.requires_grad = True
            else:
                p.requires_grad = False


    def unfreeze_layers(self, ignore=[]):
        '''
        unfreeze all layers except for those in the ignore list
        '''
        for name, p in self.model.named_parameters():
            if any([i in name for i in ignore]):
                p.requires_grad = False
            else:
                p.requires_grad = True


    def set_train_data(self, X_train, y_train):
        self.gen_params['to_augment'] = True
        self.train_data = DatasetGenerator(X_train, y_train, **self.gen_params)
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)


    def set_val_data(self, X_val, y_val):
        self.gen_params['to_augment'] = False
        self.val_data = DatasetGenerator(X_val, y_val, **self.gen_params)
        self.val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)


    def set_test_data(self, X_test, y_test):
        self.gen_params['to_augment'] = False
        self.test_data = DatasetGenerator(X_test, y_test, **self.gen_params)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)


    def train(self, data_to_use='train'):
        if data_to_use == 'train':
            dataloader = self.train_dataloader
        elif data_to_use == 'test':
            dataloader = self.test_dataloader
        elif data_to_use == 'val':
            dataloader = self.val_dataloader

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.train()
        if self.spottune:
            self.agent.train()
        total_loss = 0.
        
        for batch, (X, y) in enumerate(dataloader):
            #torch.manual_seed(self.seed)
            X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)

            if self.spottune:
                probs = self.agent(X)
                action = self.gumbel_softmax(probs.view(probs.size(0), -1, 2), self.gumbel_temperature)
                self.policy = action[:,:,1]

            #Compute prediction error
            pred = self.model(X, self.policy)
            loss = self.loss_fn(pred, y)
            total_loss += self.loss_fn(pred, y)
            
            acc = self.acc_fn(pred, y)
            auc = self.auc_fn(pred, y)
        
            #Backpropagate
            self.optimizer.zero_grad()
            if self.spottune:
                self.agent_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.spottune:
                self.agent_optimizer.step()
            
            print(f"[{batch+1:02g}/{num_batches}][{'='*int((100*((batch+1)/num_batches))//5) + '.'*int((100*((num_batches-(batch+1))/num_batches))//5)}] "
                  f"loss: {loss:>0.4f}, "
                  f"Acc: {self.acc_fn.compute():>0.4f}, "
                  f"AUC: {self.auc_fn.compute():>0.4f}", end='\r')

        iacc = self.acc_fn.compute()
        iauc = self.auc_fn.compute()
        total_loss /= num_batches

        print(f"\nAvg Train Loss: {total_loss:>0.4f}; "
              f"Total Train ACC: {iacc:>0.4f}; "
              f"Total Train AUC: {iauc:>0.4f}")    
        self.writer.add_scalars('Loss', {f"{data_to_use}_loss": total_loss}, self.epoch)
        self.writer.add_scalars('ACC', {f"{data_to_use}_acc": iacc}, self.epoch)
        self.writer.add_scalars('AUC', {f"{data_to_use}_auc": iauc}, self.epoch)

        self.acc_fn.reset()
        self.auc_fn.reset()



    def test(self, data_to_use='test'):
        if data_to_use == 'val':
            dataloader = self.val_dataloader
        elif data_to_use == 'test':
            dataloader = self.test_dataloader
        elif data_to_use == 'train':
            dataloader = self.train_dataloader

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        if self.spottune:
            self.agent.eval()
        test_loss = 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)

                if self.spottune:
                    probs = self.agent(X)
                    action = self.gumbel_softmax(probs.view(probs.size(0), -1, 2), self.gumbel_temperature)
                    self.policy = action[:,:,1]

                pred = self.model(X, self.policy)
                test_loss += self.loss_fn(pred, y)
                acc = self.acc_fn(pred, y)
                auc = self.auc_fn(pred, y)
                
                print(f"[{batch+1}/{num_batches}][{'='*int((100*((batch+1)/num_batches))//5) + '.'*int((100*((num_batches-(batch+1))/num_batches))//5)}]", end='\r')
                #print(f"[{batch+1}/{num_batches}][{'='*(batch+1) + '.'*(num_batches-(batch+1))}]", end='\r')
                      
        iacc = self.acc_fn.compute()
        iauc = self.auc_fn.compute()

        iacc_err = torch.sqrt((1/size)*iacc*(1-iacc))
        iauc_err = torch.sqrt((1/size)*iauc*(1-iauc))

        test_loss /= num_batches
        print(f"\nAvg {data_to_use.capitalize()}   Loss: {test_loss:>0.3f}; "
              f"      {data_to_use.capitalize()} ACC: {iacc:>0.3f} \u00B1 {iacc_err:>0.2}; "
              f"        {data_to_use.capitalize()} AUC: {iauc:>0.3f} \u00B1 {iauc_err:>0.2}")

  
        self.writer.add_scalars('Loss', {f"{data_to_use}_loss": test_loss}, self.epoch)
        self.writer.add_scalars('ACC', {f"{data_to_use}_acc": iacc}, self.epoch)
        self.writer.add_scalars('AUC', {f"{data_to_use}_auc": iauc}, self.epoch)
        #self.writer.add_scalars('ACC', {f"{data_to_use}_pos_err_acc": iacc + iacc_err}, self.epoch)
        #self.writer.add_scalars('ACC', {f"{data_to_use}_neg_err_acc": iacc - iacc_err}, self.epoch)
        #self.writer.add_scalars('AUC', {f"{data_to_use}_pos_err_auc": iauc + iauc_err}, self.epoch)
        #self.writer.add_scalars('AUC', {f"{data_to_use}_neg_err_auc": iauc - iauc_err}, self.epoch)

        if data_to_use == 'val':
            if (iacc > self.best_acc or test_loss < self.best_loss or (iacc+iauc)>self.best_sum) and iacc > 0.6:
                if (iacc+iauc) > self.best_sum:
                    self.best_sum = iacc+iauc
                if iacc > self.best_acc:
                    self.best_acc = iacc
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                out_path = os.path.join(self.log_dir, f"best_model_{self.epoch}_{test_loss:0.2f}_{iacc:>0.2f}_{iauc:>0.2f}.pth")
                if self.spottune:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'agent_state_dict': self.agent.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'agent_optimizer_state_dict': self.agent_optimizer.state_dict(),
                        'gen_params': self.gen_params,
                        'config' : self.config,
                        'epoch': self.epoch,
                        'loss': test_loss}, out_path)
                else:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'gen_params': self.gen_params,
                        'config' : self.config,
                        'epoch': self.epoch,
                        'loss': test_loss}, out_path)

        self.acc_fn.reset()
        self.auc_fn.reset()
        self.policy = None
        return test_loss.cpu().detach().numpy(), iacc.cpu().detach().numpy(), iauc.cpu().detach().numpy()

    def adjust_temp(self):
        for i, step in enumerate(self.temp_steps):
            if self.epoch >= step:
                self.gumbel_temperature = self.temp_vals[i]
        print(f"temperature status: {self.gumbel_temperature}")


    def adjust_seed(self):
        for i, step in enumerate(self.seed_steps):
            if self.epoch >= step:
                self.seed_switch = self.seed_vals[i]

        print(f"random status: {self.seed_switch}")


    def run(self):
        self.epoch = 0
        self.adjust_temp()
        self.adjust_seed()
        if self.seed_switch == 'high':
            torch.manual_seed(self.seed)
        for t in range(self.n_epochs):
            print(f"Epoch {t+1}/{self.n_epochs}")
            self.epoch = t + 1
            self.adjust_temp()
            self.adjust_seed()
            if self.seed_switch == 'mid':
                torch.manual_seed(self.seed)
            self.train('train')
            if self.seed_switch == 'mid':
                torch.manual_seed(self.seed)
            #torch.manual_seed(self.seed)
            val_loss = self.test('val')
            if self.config['lr_sched']:
                print('sched step')
                #self.lr_sched.step(val_loss)   
                self.lr_sched.step()   
                if self.spottune:
                    #self.agent_lr_sched.step(val_loss)
                    self.agent_lr_sched.step()
            print(f"-----------------------------------------------------------")
        self.epoch += 1
        print("Train Total")
        if self.seed_switch == 'mid':
            torch.manual_seed(self.seed)
        self.test('train')
        print("Val Total")
        if self.seed_switch == 'mid':
            torch.manual_seed(self.seed)
        self.test('val')
        print("Test")
        if self.seed_switch == 'mid':
            torch.manual_seed(self.seed)
        self.test('test')
        if self.spottune:
            torch.save({'model_state_dict': self.model.state_dict(),
                        'agent_state_dict': self.agent.state_dict(),
                        'gen_params': self.gen_params,
                        'config'    : self.config, 
                        'model_name': self.model.__class__.__name__}, os.path.join(self.log_dir, 'last_model.pth'))
        else:
            torch.save({'model_state_dict': self.model.state_dict(),
                        'gen_params': self.gen_params,
                        'config'    : self.config, 
                        'model_name': self.model.__class__.__name__}, os.path.join(self.log_dir, 'last_model.pth'))
        print("Done")


    def predict(self, data_to_use='test'):
        if data_to_use == 'test':
            dataloader = self.test_dataloader
        elif data_to_use == 'train':
            dataloader = self.train_dataloader
        elif data_to_use == 'tal':
            dataloader = self.val_dataloader

        self.model.eval()
        predict_fn = nn.Sigmoid()
        num_batches = len(dataloader)
        results = []
        true_val = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)

                if self.spottune:
                    probs = self.agent(X)
                    action = self.gumbel_softmax(probs.view(probs.size(0), -1, 2), self.gumbel_temperature)
                    self.policy = action[:,:,1]

                pred = self.model(X, self.policy)
                pred = predict_fn(pred)
                pred_np = pred.cpu().detach().numpy()
                for p in pred_np:
                    results.append(p)
                for t in y:
                    true_val.append(t.cpu().detach().numpy())
                print(f"[{batch+1}/{num_batches}][{'='*(batch+1) + '.'*(num_batches-(batch+1))}]", end='\r')

        return results, true_val

    def test_average(self, data_to_use='test'):
        torch.manual_seed(self.seed)
        tests = []

        for i in range(100):
            print(i+1)
            tests.append(self.test(data_to_use))

        tests = np.array(tests)

        print(f"Average loss: {np.mean(tests[:,0])} {np.std(tests[:,0])}")
        print(f"Average ACC : {np.mean(tests[:,1])} {np.std(tests[:,1])}")
        print(f"Average AUC : {np.mean(tests[:,2])} {np.std(tests[:,2])}")
        
        return tests


    def sample_gumbel(self, shape, eps=1e-20):
        if self.seed_switch == 'low':
            torch.manual_seed(self.seed)
        U = torch.cuda.FloatTensor(shape).uniform_()
        #U = torch.FloatTensor(shape).uniform_()
        return -Variable(torch.log(-torch.log(U + eps) + eps))


    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)


    def gumbel_softmax(self, logits, temperature = 5):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
