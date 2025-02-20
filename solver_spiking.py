import torch.nn.functional as F
import torch
from custom_neuron import CustomNeuron
from visualization import visualize
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
import pandas as pd
import os
from torch import optim
from model_spiking import VisionTransformer,ConvEnc
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader
import spikingjelly.timing_based.encoding as time_enc
from spikingjelly.activation_based import functional, monitor,encoding
from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    def __init__(self, args):

        self.args = args
        self.encoder =  encoding.PoissonEncoder()    
        out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.batch_size}_lr{args.lr}')
        self.writer = SummaryWriter(out_dir, purge_step=0)
        self.train_loader, self.test_loader = get_loader(args)
        if torch.cuda.is_available():
            # self.conv_encoder = ConvEnc(args).cuda()
            # self.weight_phase_enc = encoding.WeightedPhaseEncoder(args.T).cuda()
            # self.latency_enc = encoding.LatencyEncoder(self.args.T,enc_function='linear').cuda()
            self.model = nn.DataParallel(VisionTransformer(args).cuda())
            # self.gaussian_encoder = time_enc.GaussianTuning(n=1, m=12, x_min=torch.zeros(size=[1]).cuda(), x_max=torch.ones(size=[1]).cuda())
        else:
            self.model = VisionTransformer(args)
            # self.latency_enc = encoding.LatencyEncoder(self.args.T,enc_function='linear')
            # self.weight_phase_enc = encoding.WeightedPhaseEncoder(args.T)
            # self.conv_encoder = ConvEnc(args)
            # self.gaussian_encoder = encoding.GaussianTuning(n=1, m=12, x_min=torch.zeros(size=[1]), x_max=torch.ones(size=[1]))
        self.ce = nn.CrossEntropyLoss()

        self.v_monitor = monitor.AttributeMonitor('v', False, self.model, 
                                                  instance=CustomNeuron, 
                                                  function_on_attribute = self.to_cpu)
        self.v_loss_monitor = monitor.AttributeMonitor('v_loss', False, self.model, 
                                                  instance=CustomNeuron, 
                                                  function_on_attribute = self.to_cpu)
        self.spike_monitor = monitor.OutputMonitor(self.model, CustomNeuron, self.to_cpu)
        self.input_monitor = monitor.InputMonitor(self.model, CustomNeuron, self.to_cpu)
        self.spike_monitor.disable()
        self.v_monitor.disable()
        self.v_loss_monitor.disable()
        self.input_monitor.disable()
        print('--------Network--------')
        print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path,
                                                               'cifar_dvs.pt')))

    def init_exp_containers(self):
        return ['Experiment_name'],['Right and wrong']
    def single_batch_test(self, mode, df_name_spikes=None, df_name_vloss=None,df_name_weights=None):
        self.model.eval()
        actual = []
        pred = []

        cols,values = self.init_exp_containers()
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name:
                cols.append(f'{name}_mean')
                cols.append(f'{name}_std')
                values.append(torch.mean(param.data.cpu()).item())
                values.append(torch.std(param.data.cpu()).item())
        
        df_weights = self.update_data(df_name_weights, cols, values)
        cols,values = self.init_exp_containers()

        v_per_layer = []
        spikes_per_layer = []
        inputs_per_layer = []
        v_loss_per_layer = []
        if mode == 'single':
            batch = [x for x in next(iter(self.test_loader))]
            img_label = [(batch[0],batch[1])]
        elif mode == 'full':
            img_label = self.test_loader

        for (image, label) in img_label:
            self.spike_monitor.enable()
            self.v_monitor.enable()
            self.input_monitor.enable()
            self.v_loss_monitor.enable()
            im = image.view(28,28)
            #save_image(im, 'img1.png')
            if torch.cuda.is_available():
                    image, label = image.cuda(), label.cuda()
                    #print(label)

            with torch.no_grad():
                out_fr = 0.
                # image = torch.clip(image,0,(1-2**(-self.args.T))) # WPE
                # image = image.view(image.shape[0], -1).unsqueeze(1) # [batch_size, 1, 784]
                # encoded_img = self.gaussian_encoder.encode(image, self.args.T)  # [batch_size, 1, 784, m]
                # encoded_img = encoded_img.view(encoded_img.shape[0], -1 ,encoded_img.shape[-1])  # [batch_size, m, 784]
                # encoded_img = encoded_img.view(encoded_img.shape[0], -1 ,28,28)  # [batch_size, m, 28,28]
                # encoded_img = self.conv_encoder(image) #conv
                for _ in range(self.args.T):    
                    encoded_img = self.encoder(image) # poisson
                    # encoded_img = self.latency_enc(image)
                    # encoded_img = self.weight_phase_enc(image)
                    out_fr += self.model(encoded_img)
                # functional.reset_net(self.conv_encoder) 
                # functional.reset_net(self.weight_phase_enc)
                # functional.reset_net(self.latency_enc) 
                out_fr = out_fr / self.args.T
            functional.reset_net(self.model)
            predicted = out_fr.argmax(1)
            actual += label.tolist()
            pred += predicted.tolist()
        if self.spike_monitor.is_enable():
            spikes_per_layer = [(name, self.spike_monitor[name]) 
                    for name in self.spike_monitor.monitored_layers]
            self.spike_monitor.clear_recorded_data()
            self.spike_monitor.disable()
        if self.input_monitor.is_enable():
            inputs_per_layer = [(name, self.input_monitor[name]) 
                    for name in self.input_monitor.monitored_layers]
            self.input_monitor.clear_recorded_data()
            self.input_monitor.disable()
        if self.v_monitor.is_enable():
            v_per_layer = [(name, self.v_monitor[name]) 
                        for name in self.v_monitor.monitored_layers]
            self.v_monitor.clear_recorded_data()
            self.v_monitor.disable()
        
        if self.v_loss_monitor.is_enable():
            v_loss_per_layer = [(name, self.v_loss_monitor[name]) 
                        for name in self.v_loss_monitor.monitored_layers]
            self.v_loss_monitor.clear_recorded_data()
            self.v_loss_monitor.disable()

        if self.args.visualize:
            vs_dir = os.path.join(self.args.out_dir, 'visualization')
            if not vs_dir:
                os.mkdir(vs_dir)
            to_pil_img = torchvision.transforms.ToPILImage()
            for (image,label) in img_label:
                for i in range(len(image)):
                    to_pil_img(image[i]).save(os.path.join(vs_dir, f'input_{label[i]}_{i}.png'))
            visualize(self.args,spikes_per_layer,'output')
            visualize(self.args,inputs_per_layer,'input')
        
        
        for (layer, tensor_list) in inputs_per_layer:
            input_batch_count = len(tensor_list)
            neg_count = sum([torch.sum(torch.lt(tensor,0).int()) for tensor in tensor_list])
            b,s,e = [*tensor_list[0].shape, *([1] * (3 - len(tensor_list[0].shape)))]
            all_neg_value = sum([torch.sum(self.pos_to_0(tensor)) for tensor in tensor_list])

            cols.append(f"{layer}_total_inputs")
            cols.append(f'{layer}_average_input_negativity')
            cols.append(f'{layer}_neg_input_percent')
            values.append(input_batch_count*b*s*e)
            values.append((all_neg_value/neg_count).item())
            values.append(((neg_count/(input_batch_count*b*s*e))*100).item())
            
            
        for (layer, tensor_list) in spikes_per_layer:
            spike_batch_count = len(tensor_list)
            b,s,e = [*tensor_list[0].shape, *([1] * (3 - len(tensor_list[0].shape)))]
            
            spike_count = sum([torch.sum(tensor) for tensor in tensor_list])
            cols.append(f"{layer}_dim_bse")
            cols.append(f'{layer}_spike_count')
            cols.append(f'{layer}_total_spikes')
            cols.append(f'{layer}_spike_percent')
            values.append(f"{b},{s},{e}")
            values.append(spike_count.item())
            values.append(spike_batch_count*b*s*e)
            values.append(((spike_count/(spike_batch_count*b*s*e))*100).item())
           
        for (layer, tensor_list) in v_per_layer:
            v_batch_count = len(tensor_list)
            neg_count = sum([torch.sum(torch.lt(tensor,0).int()) for tensor in tensor_list])
            b,s,e = [*tensor_list[0].shape, *([1] * (3 - len(tensor_list[0].shape)))]
            all_neg_pot_value = sum([torch.sum(self.pos_to_0(tensor)) for tensor in tensor_list])

            cols.append(f'{layer}_total_pots')
            cols.append(f'{layer}_average_negativity')
            cols.append(f'{layer}_neg_percent')
            values.append(v_batch_count*b*s*e)
            values.append((all_neg_pot_value/neg_count).item())
            values.append(((neg_count/(v_batch_count*b*s*e))*100).item())
        
        df_spikes= self.update_data(df_name_spikes, cols, values)
        cols,values = self.init_exp_containers()

        for (layer, tensor_list) in v_loss_per_layer:
            for t in range(len(tensor_list)):
                self.writer.add_scalar(f'{layer}_v_loss',tensor_list[t],t+1)
                cols.append(f'{layer}_vloss_ts_{t+1}')
                values.append(tensor_list[t].item())
                
        df_vloss = self.update_data(df_name_vloss, cols, values)
        cols,values = self.init_exp_containers()
        df_spikes.to_csv('spike_metrics_rnw.csv',index=False)
        df_vloss.to_csv('potential_loss_max_rnw.csv',index=False)
        df_weights.to_csv('network_weights_norm_rnw.csv',index=False)
        return 0

    def update_data(self, name, cols, values):
        if name is None:
            df = pd.DataFrame([values],columns=cols)
        else:
            df = pd.read_csv(name)
            df.loc[len(df)] = values
        return df

    def pos_to_0(self,t):
            t[t > 0] = 0
            return t
    
    def test_dataset(self, db='test'):
        self.model.eval()
        
        actual = []
        pred = []

        if db.lower() == 'train':
            loader = self.train_loader
        else:
            self.spike_monitor.enable()
            self.v_monitor.enable()
            loader = self.test_loader
        
        test_loss = 0
        test_samples = 0
        spikes_per_layer=[]
        v_per_layer=[]
        for (imgs, labels) in loader:
            labels_onehot = F.one_hot(labels, self.args.n_classes).float()
            if torch.cuda.is_available():
                imgs, labels_onehot,labels = imgs.cuda(), labels_onehot.cuda(),labels.cuda()
            with torch.no_grad():
                out_fr = 0.
                imgs = imgs.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W] #dvsg
                # imgs = torch.clip(imgs,0,(1-2**(-self.args.T))) # WPE
                # imgs = imgs.view(imgs.shape[0], -1).unsqueeze(1) # [batch_size, 1, 784]
                # encoded_img = self.gaussian_encoder.encode(imgs, self.args.T)  # [batch_size, 1, 784, m]
                # encoded_img = encoded_img.view(encoded_img.shape[0], -1 ,encoded_img.shape[-1])  # [batch_size, m, 784]
                # encoded_img = encoded_img.view(encoded_img.shape[0], -1 ,28,28)  # [batch_size, m, 28,28]
                for t in range(self.args.T):    
                    encoded_img = imgs[t] #dvsg
                    # encoded_img = self.encoder(imgs) # poisson
                    # encoded_img = self.latency_enc(imgs)
                    # encoded_img = self.weight_phase_enc(imgs)
                    out_fr += self.model(encoded_img)
                # functional.reset_net(self.weight_phase_enc)
                # functional.reset_net(self.latency_enc) 
                out_fr = out_fr / self.args.T
                loss = F.mse_loss(out_fr, labels_onehot)
            test_samples += labels.numel()
            test_loss += loss.item() * labels.numel()
            predicted = out_fr.argmax(1)
            functional.reset_net(self.model)

            actual += labels.tolist()
            pred += predicted.tolist()
        test_loss /= test_samples
        if self.spike_monitor.is_enable():
            spikes_per_layer = [(name, self.spike_monitor[name]) 
                      for name in self.spike_monitor.monitored_layers]
            self.spike_monitor.clear_recorded_data()
            self.spike_monitor.disable()
        if self.v_monitor.is_enable():
            v_per_layer = [(name, self.v_monitor[name]) 
                        for name in self.v_monitor.monitored_layers]
            self.v_monitor.clear_recorded_data()
            self.v_monitor.disable()
            
        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))
        return acc, cm, test_loss,spikes_per_layer, v_per_layer

    def test(self):
        train_acc, cm,_ ,_,_= self.test_dataset('train')
        print("Tr Acc: %.2f" % train_acc)
        print(cm)

        test_acc, cm,_,_ ,_= self.test_dataset('test')
        print("Te Acc: %.2f" % test_acc)
        print(cm)

        return train_acc, test_acc

    def to_cpu(self, s_seq: torch.Tensor):
        # s_seq.shape = [T, N, *]
        return s_seq.to(torch.device("cpu"))

    def train(self):
        iter_per_epoch = len(self.train_loader)
        

        optimizer = optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs*self.args.T, verbose=True)

        for epoch in range(self.args.epochs):
            
            train_loss = 0
            train_acc = 0
            train_samples = 0
            self.model.train()

            for i, (imgs, labels) in enumerate(self.train_loader):
                labels_onehot = F.one_hot(labels, self.args.n_classes).float()
                if torch.cuda.is_available():
                    imgs, labels_onehot, labels = imgs.cuda(), labels_onehot.cuda(),labels.cuda()
                out_fr = 0.
                imgs = imgs.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W] #dvsg
            
                # imgs = torch.clip(imgs,0,(1-2**(-self.args.T))) # WPE
                # imgs = imgs.view(imgs.shape[0], -1).unsqueeze(1) # [batch_size, 1, 784]
                # encoded_img = self.gaussian_encoder.encode(imgs, self.args.T)  # [batch_size, 1, 784, m]
                # encoded_img = encoded_img.view(encoded_img.shape[0], -1 ,28,28)  # [batch_size, m, 28,28]
                for t in range(self.args.T):
                    encoded_img = imgs[t] #dvsg
                    # encoded_img = self.encoder(imgs) # poisson
                    # encoded_img = self.latency_enc(imgs)
                    # encoded_img = self.weight_phase_enc(imgs)
                    out_fr += self.model(encoded_img)
                # functional.reset_net(self.weight_phase_enc)
                # functional.reset_net(self.latency_enc) 
                out_fr = out_fr / self.args.T
                loss = F.mse_loss(out_fr, labels_onehot)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (epoch + 1, self.args.epochs, i + 1, iter_per_epoch, loss))
                
                train_samples += labels.numel()
                train_loss += loss.item() * labels.numel()
                train_acc += (out_fr.argmax(1) == labels).float().sum().item()
                functional.reset_net(self.model)


            train_loss /= train_samples
            train_acc /= train_samples
            test_acc, cm, test_loss, spikes_per_layer,v_per_layer= self.test_dataset('test')
            spike_batch_count = len(spikes_per_layer[0][1])
            v_batch_count = len(v_per_layer[0][1])
            
            for (layer, tensor_list) in spikes_per_layer:
                b,s,e = [*tensor_list[0].shape, *([1] * (3 - len(tensor_list[0].shape)))]
                spike_count = sum([torch.sum(tensor) for tensor in tensor_list])
                self.writer.add_scalar(f'spike_percent_{layer}',
                                       (spike_count/(spike_batch_count*b*s*e))*100 , epoch)
            for (layer, tensor_list) in v_per_layer:
                neg_count = sum([torch.sum(torch.lt(tensor,0).int()) for tensor in tensor_list])
                b,s,e = [*tensor_list[0].shape, *([1] * (3 - len(tensor_list[0].shape)))]
                all_neg_pot_value = sum([torch.sum(self.pos_to_0(tensor)) for tensor in tensor_list])

                self.writer.add_scalar(f'mean_negativity_{layer}', all_neg_pot_value/neg_count, epoch)
                self.writer.add_scalar(f'neg_v_percent_{layer}', 
                                       (neg_count/(v_batch_count*b*s*e))*100, epoch)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('train_acc', train_acc, epoch)
            self.writer.add_scalar('test_loss', test_loss, epoch)
            self.writer.add_scalar('test_acc', test_acc, epoch)
            for name, weight in self.model.named_parameters():
                self.writer.add_histogram(name,weight, epoch)    
                self.writer.add_histogram(f'{name}.grad',weight.grad, epoch)

            print("Test acc: %0.2f" % test_acc)
            print(cm, "\n")
            cos_decay.step()
        torch.save(self.model.state_dict(), os.path.join(self.args.model_path, 'cifar_dvs.pt'))