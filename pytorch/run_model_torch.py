import os
from datetime import datetime
from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchmetrics

from gbm_project.pytorch.dataset_class import DatasetGenerator
from gbm_project.pytorch.resnet_torch import ResNet3D18, ResNet3D50

from MedicalNet.models import resnet

MODELS = ['ResNet18',
          'ResNet50']
TRANSFER_MODELS = ['MedResNet18',
                   'MedResNet50',
                   'MedResNet101',
                   'ResNet50_torch']

class RunModel(object):
    def __init__(self, config, gen_params):
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']
        self.n_channels = gen_params['n_channels']
        self.gen_params = gen_params
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.acc_fn = torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(self.device)
        self.auc_fn = torchmetrics.classification.BinaryAUROC().to(self.device)

        self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f"logs are located at: {self.logdir}")
        self.writer = SummaryWriter(self.log_dir)
        print('remember to set the data')
        self.epoch = 0
        self.best_acc =0.


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

###########################################################################################################
            else:
                print("no transfer model chosen, choose from:")
                print(f"{TRANSFER_MODELS}")
             
        # set the optimizer here, since it needs the model parameters in its initialization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['l2_reg'])
        self.lr_sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_step_size'], gamma=self.config['lr_gamma'])


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
        self.val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True)


    def set_test_data(self, X_test, y_test):
        self.gen_params['to_augment'] = False
        self.test_data = DatasetGenerator(X_test, y_test, **self.gen_params)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True)


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
        total_loss = 0.

        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
            
            #Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            total_loss += self.loss_fn(pred, y)
            
            acc = self.acc_fn(pred, y)
            auc = self.auc_fn(pred, y)
        
            #Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print(f"[{batch+1}/{num_batches}][{'='*int((100*((batch+1)/num_batches))//5) + '.'*int((100*((num_batches-(batch+1))/num_batches))//5)}] "
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
        test_loss = 0
        
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y)
                acc = self.acc_fn(pred, y)
                auc = self.auc_fn(pred, y)
                
                print(f"[{batch+1}/{num_batches}][{'='*int((100*((batch+1)/num_batches))//5) + '.'*int((100*((num_batches-(batch+1))/num_batches))//5)}]", end='\r')
                #print(f"[{batch+1}/{num_batches}][{'='*(batch+1) + '.'*(num_batches-(batch+1))}]", end='\r')
                      
        iacc = self.acc_fn.compute()
        iauc = self.auc_fn.compute()
        test_loss /= num_batches
        print(f"\nAvg {data_to_use.capitalize()}   Loss: {test_loss:>0.4f}; "
              f"      {data_to_use.capitalize()} ACC: {iacc:>0.4f}; "
              f"        {data_to_use.capitalize()} AUC: {iauc:>0.4f}")

  
        self.writer.add_scalars('Loss', {f"{data_to_use}_loss": test_loss}, self.epoch)
        self.writer.add_scalars('ACC', {f"{data_to_use}_acc": iacc}, self.epoch)
        self.writer.add_scalars('AUC', {f"{data_to_use}_auc": iauc}, self.epoch)

        if data_to_use == 'val':
            if iacc > self.best_acc:
                self.best_acc = iacc
                out_path = os.path.join(self.log_dir, f"best_model_{self.epoch}_{test_loss:0.2f}_{iacc:>0.2f}_{iauc:>0.2f}.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    'loss': test_loss}, out_path)

        self.acc_fn.reset()
        self.auc_fn.reset()


    def run(self):
        self.epoch = 0
        for t in range(self.n_epochs):
            print(f"Epoch {t+1}/{self.n_epochs}")
            self.epoch = t + 1
            self.train('train')
            self.test('val')
            #self.lr_sched.step()   
            print(f"-----------------------------------------------------------")
        self.epoch += 1
        print("Train Total")
        self.test('train')
        print("Val Total")
        self.test('val')
        print("Test")
        self.test('test')
        torch.save({'state_dict': self.model.state_dict(),
                    'gen_params': self.gen_params,
                    'config'    : self.config}, os.path.join(self.log_dir, 'last_model.pth'))
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

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                pred = self.model(X)
                pred = predict_fn(pred)
                pred_np = pred.cpu().detach().numpy()
                for p in pred_np:
                    results.append(p)
                print(f"[{batch+1}/{num_batches}][{'='*(batch+1) + '.'*(num_batches-(batch+1))}]", end='\r')

        return results
