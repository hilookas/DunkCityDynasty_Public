import time
import random
import numpy as np

from baselines.env import Env
import json

from baselines.wrappers import RLWrapper

def main():
    # env config
    # --- win env
    # config = {
    #     'id': 1,
    #     'env_setting': 'win',
    # }

    # --- linux env
    config = {
        'id': 21,
        'env_setting': 'linux',
        'render': True,
    }

    # # --- multi_machine
    # config = {
    #     'id': 1,
    #     'env_setting': 'multi_machine',
    #     'machine_server_ip': '10.219.204.76',
    #     'machine_server_port': 6667,
    # }

    env = Env(config)
    while True:
        agent_id, raw_ob = env.reset()
        while True:
            print(agent_id, raw_ob[1]['self_state']['character_id'], raw_ob[1]['self_state']['position_type'], raw_ob[2])
            action_in = np.random.randint(0, 52)
            agent_id, raw_ob, truncated, no_time = env.step(agent_id, action_in)
            print(truncated, no_time)

if __name__ == '__main__':
    main()