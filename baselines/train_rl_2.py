from argparse import Namespace
import queue
import random
import sys,os;sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import time
from model import Model
from wrappers import RLWrapper
from env import Env
import torch.multiprocessing as mp
from datetime import datetime
from typing import List
import itertools
import traceback

def collect(id, args, run_name, model_queue: mp.Queue, data_queue: mp.Queue):
    def tprint(*args):
        print(f"[{datetime.now().isoformat()}] collector_{id}: ", *args)
    tprint("started")

    device = torch.device("cpu")

    wrapper = RLWrapper()
    env = Env({ 'env_setting': 'linux', 'id': id, 'render': id == 0 }, tprint=tprint) # render only id == 0

    model_state_dict, value_model_state_dict = model_queue.get()

    model = Model()
    model.load_state_dict(model_state_dict)
    model.to(device)
    tprint("model loaded")

    value_model = Model()
    value_model.load_state_dict(value_model_state_dict)
    value_model.to(device)
    tprint("value_model loaded")
    
    oppo_model = Model()

    oppo_model_load_paths = [
        ('output_bc2_epoch2_20231029/bc_model_1', 1), # 0
        ('output_bc2_epoch10_20231030/bc_model_9', 1), # 1
        ('output_rl2_1701111766_better_than_bc/output_rl2_1701111766_better_than_bc/model_3907911', 1), # 2
        ('output_rl2_1701183305_start_better_than_bc_rl/output_rl2_1701199168_start_better_than_bc_rl_resume/model_4909562', 2), # 3
        ('output_official_rl_baseline/rl_model', 1), # 4
        ('output_rl2_1701240946_oppo_to_all/output_rl2_1701240946/model_2004594', 2), # 5
        ('output_rl2_1701240946_oppo_to_all/output_rl2_1701280492/model_5112162', 3), # 6
    ]

    time.sleep(id * 10)

    for game_i in itertools.count(): # one complete episode (with multiple truncated episode)
        try:
            tprint(f"game {game_i}")

            # clear queue and get the last model
            tprint(f"model_queue_size {model_queue.qsize()}")
            model_state_dicts = None
            try:
                while True:
                    model_state_dicts = model_queue.get_nowait()
            except queue.Empty as e:
                pass
            tprint(f"after model_queue_size {model_queue.qsize()}")
            if model_state_dicts is not None:
                model_state_dict, value_model_state_dict = model_state_dicts

                if model_state_dict is not None:
                    model.load_state_dict(model_state_dict)
                    model.to(device)
                    tprint("model reloaded")

                if value_model_state_dict is not None:
                    value_model.load_state_dict(value_model_state_dict)
                    value_model.to(device)
                    tprint("value_model reloaded")
            
            paths, weights = list(zip(*oppo_model_load_paths)) # unzip
            oppo_model_id, oppo_model_load_path = random.choices(list(enumerate(paths)), weights, k=1)[0]

            tprint(f"use oppo_model_{oppo_model_id}: {oppo_model_load_path}")
            
            oppo_model.load_state_dict(torch.load(oppo_model_load_path, map_location=device))
            oppo_model.to(device)

            tprint(f"reseting")
            agent_id, raw_ob = env.reset()
            tprint(f"reseted")
            truncated = False
            done_count = 0
            last_step_time = time.time()
        
            ma_partial_exp = {agent_id: None for agent_id in range(3)}
            ma_exps = {agent_id: Namespace(obs=[], actions=[], log_probs=[], values=[], rewards=[], reward_details=[], raw_obs=[]) for agent_id in range(3)}

            while True: # one step
                # tprint(f"step")
                infer_start_time = time.time()
                with torch.no_grad():
                    # get action
                    ob = wrapper.states_wrapper(raw_ob)
                    ob = torch.tensor(np.concatenate([ob[i] for i in range(8)]), dtype=torch.float32).to(device).unsqueeze(0) # shape: [1, 520]
                    assert ob.shape[1] == 30+73+73+73+73+73+73+52
                    state_input = torch.split(ob, [30,73,73,73,73,73,73,52], dim=1)
                    if agent_id < 3: # 3, 4, 5 为 oppo
                        _, probs = model(state_input) # value shape: [1, 1] # TODO 把经过wrapper处理的observation数据整合在一起的大向量，完整扔到模型中处理
                        value, _ = value_model(state_input) # value shape: [1, 1] # TODO 把经过wrapper处理的observation数据整合在一起的大向量，完整扔到模型中处理
                        value = value.reshape(-1) # flatten
                    else:
                        _, probs = oppo_model(state_input)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample() # shape: [1]
                    log_prob = dist.log_prob(action) # shape: [1] # 对6个不同的agent处理都是一样的，分别单独输入到模型中
                    action_in = action.cpu().numpy().item()

                    if agent_id < 3: # 3, 4, 5 为 oppo
                        if ma_partial_exp[agent_id] is not None:
                            tprint(f"agent {agent_id} receive another observation, override")
                        partial_exp = Namespace()
                        partial_exp.ob = ob
                        partial_exp.action = action
                        partial_exp.log_prob = log_prob
                        partial_exp.value = value
                        ma_partial_exp[agent_id] = partial_exp
                    
                # 场景内6个agent应该都是我们控制的，每次step后，会返回6个agent中间其中一个的agent的数据。
                # 看起来这里的代码会将每个agent都视为单独的情况然后扔到模型中训练
                step_start_time = time.time()
                agent_id, raw_ob, truncated, no_time = env.step(agent_id, action_in) # ! raise exceptions
                step_time = time.time()

                iter_time = int((step_time - last_step_time)*1000)
                other_time = int((infer_start_time - last_step_time)*1000)
                infer_time = int((step_start_time - infer_start_time)*1000)
                env_time = int((step_time - step_start_time)*1000)
                if iter_time > 500:
                    tprint(f"lag: iter_time: {iter_time}ms = other_time: {other_time}ms + infer_time: {infer_time}ms + env_time: {env_time}ms")

                last_step_time = step_time

                if agent_id < 3: # 3, 4, 5 为 oppo
                    if ma_partial_exp[agent_id] is None:
                        tprint(f"agent {agent_id} receive an reward while no action to env, ignore")
                    else:
                        ma_exps[agent_id].obs.append(ma_partial_exp[agent_id].ob) # torch.Size([1, 520])
                        ma_exps[agent_id].actions.append(ma_partial_exp[agent_id].action) # torch.Size([1])
                        ma_exps[agent_id].log_probs.append(ma_partial_exp[agent_id].log_prob) # torch.Size([1])
                        ma_exps[agent_id].values.append(ma_partial_exp[agent_id].value)
                        reward = wrapper.rewards_wrapper(raw_ob)
                        # torch will auto infer type from data in, so we need manually set data type avoiding 
                        ma_exps[agent_id].rewards.append(torch.tensor([reward["tot"]], dtype=torch.float32)) # torch.Size([1])
                        ma_exps[agent_id].reward_details.append(torch.tensor([[reward[key] for key in sorted(RLWrapper.rewards_default().keys())]], dtype=torch.float32)) # torch.Size([1, 29])
                        ma_partial_exp[agent_id] = None

                    if truncated: # one truncated episode finished
                        tprint(f"truncated episode finished")
                        
                        exp = Namespace()
                        exp.obs = torch.cat(ma_exps[agent_id].obs)
                        exp.actions = torch.cat(ma_exps[agent_id].actions)
                        exp.log_probs = torch.cat(ma_exps[agent_id].log_probs)
                        exp.values = torch.cat(ma_exps[agent_id].values)
                        exp.rewards = torch.cat(ma_exps[agent_id].rewards)
                        exp.reward_details = torch.cat(ma_exps[agent_id].reward_details)
                        data_queue.put_nowait((id, agent_id, exp, oppo_model_id))
                        ma_exps[agent_id] = Namespace(obs=[], actions=[], log_probs=[], values=[], rewards=[], reward_details=[], raw_obs=[])

                        if no_time:
                            done_count += 1
                        if done_count == 3:
                            tprint("done")
                            break
        except Exception as e:
            tprint(f"ignored exception @collect: {traceback.format_exc()}")
def train():
    def tprint(*args):
        print(f"[{datetime.now().isoformat()}] train: ", *args)
    tprint("started")

    args = Namespace()
    run_name = f"rl2_{int(time.time())}"

    writer = SummaryWriter(f'./output_{run_name}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init envs
    args.num_envs = 30 # the number of parallel game environments # too many envs may cause insufficient memory
    model_queues: List[mp.Queue] = []
    data_queues: List[mp.Queue] = []
    collects: List[mp.Process] = []
    mp.set_start_method('spawn', force=True)
    for id in range(args.num_envs):
        model_queue = mp.Queue()
        data_queue = mp.Queue()
        p = mp.Process(target=collect, args=(id, args, run_name, model_queue, data_queue))
        p.start()
        tprint(f"collector_{id}: pid: {p.pid}")
        model_queues.append(model_queue)
        data_queues.append(data_queue)
        collects.append(p)

    global_step = 0 # fresh run
    # global_step = 7618051 # resume
    # load_dir = 'output_rl2_1701331406' # resume

    model = Model()
    if global_step == 0:
        model.load_state_dict(torch.load(f"output_rl2_1701183305_start_better_than_bc_rl/output_rl2_1701199168_start_better_than_bc_rl_resume/model_4909562")) # fresh run
    else:
        model.load_state_dict(torch.load(f"{load_dir}/model_{global_step}")) # resume
    model.to(device) # for Module, to() happens in place 

    value_model = Model()
    if global_step == 0:
        value_model.load_state_dict(torch.load(f"output_rl2_1701183305_start_better_than_bc_rl/output_rl2_1701199168_start_better_than_bc_rl_resume/val_model_4909562")) # fresh run
    else:
        value_model.load_state_dict(torch.load(f"{load_dir}/val_model_{global_step}")) # resume
    value_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    if global_step == 0:
        optimizer.load_state_dict(torch.load(f"output_rl2_1701183305_start_better_than_bc_rl/output_rl2_1701199168_start_better_than_bc_rl_resume/opt_4909562")) # fresh run
    else:
        optimizer.load_state_dict(torch.load(f"{load_dir}/opt_{global_step}")) # resume

    value_optimizer = optim.Adam(value_model.parameters(), lr=1e-4, eps=1e-5)
    if global_step == 0:
        value_optimizer.load_state_dict(torch.load(f"output_rl2_1701183305_start_better_than_bc_rl/output_rl2_1701199168_start_better_than_bc_rl_resume/val_opt_4909562")) # fresh run
    else:
        value_optimizer.load_state_dict(torch.load(f"{load_dir}/val_opt_{global_step}")) # resume

    tprint("model loaded")

    # send infer model out
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    value_model_cpu = {k: v.cpu() for k, v in value_model.state_dict().items()}
    for model_queue in model_queues:
        model_queue.put((model_cpu, value_model_cpu))

    last_save_step = global_step
    
    last_time = time.time()
    last_step = global_step

    while True:
        # collect data
        exps = []
        while not exps: # exps is empty
            for collector_id, data_queue in enumerate(data_queues):
                show_queue_size = data_queue.qsize() != 0
                if show_queue_size: 
                    tprint(f"data_queue_size collector_{collector_id} {data_queue.qsize()}")
                try:
                    while True:
                        collector_id, agent_id, exp, oppo_model_id = data_queue.get_nowait()
                        exps.append(exp)
                        ep_len = exp.obs.shape[0]
                        ep_ret = exp.rewards.sum()
                        global_step += ep_len
                        tprint(f"collector_{collector_id} agent_{agent_id} oppo_model_{oppo_model_id} episode length: {ep_len}")
                        tprint(f"collector_{collector_id} agent_{agent_id} oppo_model_{oppo_model_id} episode return: {ep_ret}")
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)
                        writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                        writer.add_scalar(f"charts/episodic_length_oppo_model_{oppo_model_id}", ep_len, global_step)
                        writer.add_scalar(f"charts/episodic_return_oppo_model_{oppo_model_id}", ep_ret, global_step)
                        for i, key in enumerate(sorted(RLWrapper.rewards_default().keys())):
                            sum = 0
                            for step in range(ep_len): # 记录各种rewards
                                sum += exp.reward_details[step][i]
                            writer.add_scalar("rewards/" + key, sum, global_step)
                        tprint(f"collected trajectorys: {len(exps)}")
                except queue.Empty as e:
                    pass
                if show_queue_size: 
                    tprint(f"after data_queue_size collector_{collector_id} {data_queue.qsize()}")
            time.sleep(1)
        tprint(f"collected finish, trajectory_num: {len(exps)}")
        writer.add_scalar('charts/trajectory_num', len(exps), global_step)

        for exp in exps:
            ep_len = exp.obs.shape[0]

            obs = exp.obs.to(device) # torch.Size([799, 520])
            actions = exp.actions.to(device) # torch.Size([799])
            old_log_probs = exp.log_probs.to(device) # torch.Size([799])
            old_values = exp.values.to(device) # torch.Size([799])
            rewards = exp.rewards # torch.Size([799])
            
            args.gamma = 0.99
            args.gae_lambda = 0.95
            with torch.no_grad():
                advantages = torch.zeros_like(rewards, dtype=torch.float32).to(device)
                lastgaelam = 0
                for t in reversed(range(ep_len)):
                    if t == ep_len - 1:
                        nextvalues = 0
                    else:
                        nextvalues = old_values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues - old_values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
                returns = advantages + old_values

            # update
            mb_obs, mb_actions, mb_returns, mb_old_log_probs = obs, actions, returns, old_log_probs # 基于动手学强化学习教程的观察，好像不加 minibatch 会更好

            # evaluate
            state_input = torch.split(mb_obs, [30,73,73,73,73,73,73,52], dim=1)
            _, probss = model(state_input) # new_values shape: [9, 1] probss shape: [9, 52]
            new_values, _ = value_model(state_input)
            new_values = new_values.reshape(-1)
            dist = torch.distributions.Categorical(probss)
            new_log_probs = dist.log_prob(mb_actions) # shape: [9]
            entropy = dist.entropy() # shape: [9]

            logratio = new_log_probs - mb_old_log_probs
            ratio = logratio.exp()

            args.clip_coef = 0.2
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

            mb_advantages = advantages
            # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss = -torch.min(ratio * mb_advantages, torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantages).mean()

            entropy_loss = entropy.mean()

            # Value loss
            v_loss = nn.MSELoss()(mb_returns, new_values)

            args.ent_coef = 0.01
            loss = pg_loss - args.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5) # solve NaN on output_rl2_1700497425
            optimizer.step()

            value_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), 0.5)
            value_optimizer.step()

        # log
        writer.add_scalar('losses/loss', loss.item(), global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", clipfrac.item(), global_step)
            
        current_time = time.time()
        sps = (global_step - last_step) / (current_time - last_time)
        tprint(f"step_per_second: {sps}")
        writer.add_scalar("charts/step_per_second", sps, global_step)
        last_step = global_step
        last_time = current_time

        # save_model
        if global_step - last_save_step > 100000:
            tprint(f"saved {global_step}")
            torch.save(model.state_dict(), f"./output_{run_name}/model_{global_step}")
            torch.save(optimizer.state_dict(), f"./output_{run_name}/opt_{global_step}")
            torch.save(value_model.state_dict(), f"./output_{run_name}/val_model_{global_step}")
            torch.save(value_optimizer.state_dict(), f"./output_{run_name}/val_opt_{global_step}")
            last_save_step = global_step

        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        value_model_cpu = {k: v.cpu() for k, v in value_model.state_dict().items()}
        for model_queue in model_queues: # TODO 使用rpc解决model_queue序列长的问题
            model_queue.put((model_cpu, value_model_cpu))

        tprint(f"end 1 update")

if __name__ == '__main__':
    train()