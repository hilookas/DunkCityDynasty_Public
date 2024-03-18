import sys,os;sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter 
from bc_collector import BCDataset, collate_fn
from baselines.model import Model
import numpy as np
import random
import time
from datetime import datetime
import itertools
import torch.nn.functional as F
from torch import nn

def all_seed(seed):
    ''' 设置随机种子，保证实验可复现，同时保证GPU和CPU的随机种子一致
    '''
    os.environ['PYTHONHASHSEED'] = str(seed) # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # 
    torch.cuda.manual_seed(seed) # config for GPU
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def calc_acc(tprint, test_dataset, model, device, sample_times=16):
    test_states, test_actions, _ = test_dataset.sample(sample_times)
    test_accuracy = []
    test_accuracy_no_zero = []
    for test_state, test_action in zip(test_states, test_actions):
        if len(test_action) == 0:
            tprint("len 0")
            continue 
        with torch.no_grad():
            test_state = [inner_test_state.to(device) for inner_test_state in test_state]
            test_action = test_action.to(device)
            _, test_pred_logits = model(test_state)
            test_pred_action = test_pred_logits.argmax(-1)
            test_accuracy.append((test_pred_action == test_action).float().mean().detach().cpu().numpy())
            test_accuracy_no_zero.append(torch.logical_and(test_pred_action == test_action, test_action > torch.full_like(test_action, 0)).float().mean().detach().cpu().numpy())
    test_accuracy = sum(test_accuracy) / len(test_accuracy)
    test_accuracy_no_zero = sum(test_accuracy_no_zero) / len(test_accuracy_no_zero)
    return test_accuracy, test_accuracy_no_zero

class Config:
    def __init__(self):
        self.data_path = "./full_human_data"

def train(cfg):
    # all_seed(cfg.seed)

    def tprint(*args):
        print(f"[{datetime.now().isoformat()}] train: ", *args)
    tprint("started")
    
    run_name = f"bc2_{int(time.time())}"

    writer = SummaryWriter(f'./output_{run_name}')

    device = torch.device('cuda:1')

    train_dataset = BCDataset(cfg.data_path, is_train=True) # len=354033
    test_dataset = BCDataset(cfg.data_path, is_train=False)

    # batch size不能变，否则第二个维度会变成最大的那个
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=16, collate_fn=collate_fn)

    model = Model()
    model.load_state_dict(torch.load(f"output_rl2_1701280492/model_5112162"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(torch.load(f"output_rl2_1701280492/opt_5112162"))

    test_accuracy, test_accuracy_no_zero = calc_acc(tprint, test_dataset, model, device, 100)
    print(f'test_acc: {test_accuracy}, test_acc_no_zero: {test_accuracy_no_zero}')

    global_step = 0
    for i_epoch in itertools.count(): # infinite epoch
        for i_batch, (global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action, weight) in enumerate(train_dataloader):
            global_state = global_state.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54, 30])
            self_state = self_state.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54, 73])
            ally0_state = ally0_state.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54, 73])
            ally1_state = ally1_state.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54, 73])
            enemy0_state = enemy0_state.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54, 73])
            enemy1_state = enemy1_state.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54, 73])
            enemy2_state = enemy2_state.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54, 73])
            action = action.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54])
            weight = weight.squeeze(0).to(device) # unsqueezed: torch.Size([1, 54])
            # Forward
            _, logits = model([global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state])
            loss = F.cross_entropy(logits, action).mean()
            # m = Categorical(logits=logits)
            # loss = -m.log_prob(action) * weight
            # loss = loss.mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if i_batch % 100 == 0:
                test_accuracy, test_accuracy_no_zero = calc_acc(tprint, test_dataset, model, device)
                print(f'epoch_{i_epoch} batch_{i_batch} global_step_{global_step}, loss: {loss.item()}, test_acc: {test_accuracy}, test_acc_no_zero: {test_accuracy_no_zero}')
                writer.add_scalar('loss', loss, global_step)
                writer.add_scalar('acc', test_accuracy, global_step)
                writer.add_scalar('acc_no_zero', test_accuracy_no_zero, global_step)
            
            if i_batch % 100 == 0: # 1 epoch 7 次
                torch.save(model.state_dict(), f"./output_{run_name}/model_{i_epoch}_{global_step}")
                torch.save(optimizer.state_dict(), f"./output_{run_name}/opt_{i_epoch}_{global_step}")

        torch.save(model.state_dict(), f"./output_{run_name}/model_{i_epoch}_{global_step}")
        torch.save(optimizer.state_dict(), f"./output_{run_name}/opt_{i_epoch}_{global_step}")

if __name__ == '__main__':
    cfg = Config()
    train(cfg)