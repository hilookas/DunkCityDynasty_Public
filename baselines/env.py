import os
import re
import signal
import time
import grpc
import threading
import subprocess
import queue

import baselines.multi_machine.machine_comm_pb2 as machine_comm_pb2
import baselines.multi_machine.machine_comm_pb2_grpc as machine_comm_pb2_grpc
from baselines.tcp_server import ThreadedTCPRequestHandler, ThreadedTCPServer, is_end_state

from filelock import FileLock
import traceback

class Env():
    NUM_AGENTS = 6 # total game agent

    def __init__(self, config, tprint=print):
        # env config
        self.id = config['id']
        self.env_setting = config['env_setting']
        self.client_path = 'game_package_release'
        self.game_server_ip = '123.123.123.123'
        self.game_server_port = 12312
        self.render = config['render']

        if self.env_setting =='multi_machine':
            self.machine_server_ip = config['machine_server_ip']
            self.machine_server_port = config['machine_server_port']

        if self.env_setting =='linux':
            if not self.render:
                self.xvfb_display = 10 + self.id
                cmd = f"mkdir -p xvfb_fbdir/{self.xvfb_display} && Xvfb :{self.xvfb_display} -screen 0 300x300x24 -fbdir xvfb_fbdir/{self.xvfb_display} 2>&1 | ts \'[%Y-%m-%d %H:%M:%S]\' >> logs/xvfb.{self.id}.log &"
                os.system(cmd)
                time.sleep(1)

        self.tprint = tprint
        
        self.rl_server_ip = '127.0.0.1'
        self.rl_server_port = 3330 + self.id
        
        # set tcp server
        self.tcp_server = ThreadedTCPServer((self.rl_server_ip, self.rl_server_port), ThreadedTCPRequestHandler)
        self.tcp_server.block_on_close = False
        self.tcp_server.daemon_threads = True
        self.tcp_server.tprint = self.tprint

        # start server thread
        thread = threading.Thread(target=self.tcp_server.serve_forever)
        thread.daemon = True
        thread.start()

    def _allocate_username(self):
        # allocate available user name
        avail_user_name = None
        while avail_user_name is None:
            with FileLock("accounts.txt"+'.lock'): # avoid racing condition when multiagent train
                with open("accounts.txt", 'r') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    segs = line.strip().split(" ")
                    if len(segs) == 1:
                        user_name = segs[0]
                        expires_at = 0
                    elif len(segs) == 2:
                        user_name = segs[0]
                        expires_at = int(segs[1])
                    else:
                        assert False

                    if time.time() > expires_at:
                        avail_user_name = user_name
                        expires_at = int(time.time() + 600) # 认为对局至少持续10分钟
                        del lines[i]
                        lines.append(f"{user_name} {expires_at}\n")
                        break

                with open("accounts.txt", 'w') as f:
                    f.writelines(lines)
            if avail_user_name is None:
                self.tprint("no available user_name")
                time.sleep(1)
        return avail_user_name

    def reset(self):
        """reset func
        """
        self._close_client()
        
        self.user_name = self._allocate_username()
        self.tprint(f"allocate user_name {self.user_name}")
        
        # set stream data
        self.stream_data = {
            'established': threading.Lock(),
            'state': queue.Queue(1),
            'action': queue.Queue(1)
        }
        self.tcp_server.stream_data = self.stream_data

        self._start_client()

        self.no_time = False

        # 2. get new state from client
        agent_id, state = self.stream_data['state'].get(timeout=240) # block with timeout

        # hyperparameter reset
        return agent_id, state

    def step(self, agent_id, action):
        """step func
        """
        # 1. set action
        self.stream_data['action'].put((agent_id, int(action))) # block

        # 2. get new state from client
        agent_id, state = self.stream_data['state'].get(timeout=60) # block with timeout

        truncated = is_end_state(state)
        
        if state[1]['global_state']['match_remain_time'] < 1:
            self.no_time = True

        return agent_id, state, truncated, self.no_time

    # Game Client Script

    def _start_client(self):
        """start game client
        """
        if self.env_setting == 'win':
            # run game client 
            cmd = f"{self.client_path}/Lx33.exe {self.game_server_ip} {self.game_server_port} {self.rl_server_ip} {self.rl_server_port} {self.user_name}"
            p = subprocess.Popen(cmd, shell=False)
            self.pid = p.pid

        elif self.env_setting == 'linux':
            config_file = f"{self.client_path}/Lx33_Data/boot.config"
            with FileLock(config_file+'.lock'): # avoid racing condition when multiagent train
                with open(config_file, 'r') as f:
                    lines = f.readlines()
                if not self.render:
                    lines[-1] = lines[-1].replace('0', '1')
                else:
                    lines[-1] = lines[-1].replace('1', '0')
                with open(config_file, 'w') as f:
                    f.writelines(lines)

                # run game client
                if self.render:
                    # https://forum.winehq.org/viewtopic.php?t=2255
                    # `winecfg` and toggle on 'Emulate a virtual desktop' to windowed
                    cmd = f'export DISPLAY=:0 && export STEAM_COMPAT_DATA_PATH=~/.local/share/Steam/steamapps/compatdata/dcd && export STEAM_COMPAT_CLIENT_INSTALL_PATH=~/.local/share/Steam/ && ~/.local/share/Steam/steamapps/common/Proton\\ -\\ Experimental/proton run {self.client_path}/Lx33.exe {self.game_server_ip} {self.game_server_port} {self.rl_server_ip} {self.rl_server_port} {self.user_name} 2>&1 | ts \'[%Y-%m-%d %H:%M:%S]\' >> logs/client.{self.id}.log &'
                else:
                    cmd = f'export DISPLAY=:{self.xvfb_display} && wine {self.client_path}/Lx33.exe {self.game_server_ip} {self.game_server_port} {self.rl_server_ip} {self.rl_server_port} {self.user_name} 2>&1 | ts \'[%Y-%m-%d %H:%M:%S]\' >> logs/client.{self.id}.log &'
                p = subprocess.Popen(cmd, shell=True, start_new_session=True)
                
                # get game client pid
                self.pid = os.getpgid(p.pid)

                time.sleep(8) # wait client to read config
            
        elif self.env_setting == 'multi_machine':
            with grpc.insecure_channel(f'{self.machine_server_ip}:{self.machine_server_port}') as channel:
                stub = machine_comm_pb2_grpc.ClientCommStub(channel)
                resp = stub.Cmd(machine_comm_pb2.ClientCmd(
                    client_id=self.id,
                    cmd='start_client',
                    rl_server_ip=self.rl_server_ip,
                    rl_server_port=self.rl_server_port,
                    user_name=self.user_name,
                ))

    def _close_client(self):
        """close game client
        """
        if self.env_setting == 'win':
            if hasattr(self, 'pid') and self.pid is not None:
                cmd = f"taskkill /F /PID {self.pid}"
                subprocess.call(cmd, shell=False)
                self.pid = None
        
        elif self.env_setting == 'linux':
            if hasattr(self, 'pid') and self.pid is not None:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                    time.sleep(4) # wait to kill
                except Exception as e:
                    self.tprint(f"ignored exception @_close_client: {traceback.format_exc()}")
                self.pid = None
            
            if self.render: # fix proton
                # find pid
                # TODO kill unity crash handler
                cmd = 'ps -ef|grep Lx33.exe|grep -iv proton|grep -iv steam'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                err_msg = result.stderr.strip()
                if err_msg:
                    self.tprint(f'ignored exception, cmd err during ` {cmd} `:', err_msg)

                for line in result.stdout.splitlines():
                    if f'{self.rl_server_ip} {self.rl_server_port}' in line:
                        temp = re.sub(' +', ' ', line).split(' ')
                        if len(temp) > 1:
                            self.pid = int(temp[1])

                            cmd = f"kill -9 {self.pid}"
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            err_msg = result.stderr.strip()
                            if err_msg:
                                self.tprint(f'ignored exception, cmd err during ` {cmd} `:', err_msg)

                            self.pid = None
                            
                            time.sleep(4) # wait to kill

                            break

        elif self.env_setting == 'multi_machine':
            with grpc.insecure_channel(f'{self.machine_server_ip}:{self.machine_server_port}') as channel:
                stub = machine_comm_pb2_grpc.ClientCommStub(channel)
                resp = stub.Cmd(machine_comm_pb2.ClientCmd(
                    client_id=self.id,
                    cmd='close_client',
                    rl_server_ip=self.rl_server_ip,
                    rl_server_port=self.rl_server_port,
                ))
                # if resp.msg != 'ok':
                #     raise Exception('error!!')
