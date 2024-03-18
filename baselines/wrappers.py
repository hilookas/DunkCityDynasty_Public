import numpy as np

def onehot(num, size):
    """ One-hot encoding
    """
    onehot_vec = np.zeros(size)
    if num < size:
        onehot_vec[num] = 1
    return onehot_vec.tolist()

class RLWrapper:
    def __init__(self):
        self.concated_states = False
    def states_wrapper(self, state_info):
        # more details, see: https://fuxirl.github.io/DunkCityDynasty/#/env_info
        states_dict = state_info[1]
        global_state = self.handle_global_states(states_dict['global_state'])
        self_state = self.handle_agent_states(states_dict['self_state'])
        ally0_state = self.handle_agent_states(states_dict['ally_0_state'])
        ally1_state = self.handle_agent_states(states_dict['ally_1_state'])
        enemy0_state = self.handle_agent_states(states_dict['enemy_0_state'])
        enemy1_state = self.handle_agent_states(states_dict['enemy_1_state'])
        enemy2_state = self.handle_agent_states(states_dict['enemy_2_state'])
        action_mask = np.array(state_info[2]) # legal action
        observations = [global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action_mask]
        if self.concated_states:
            observations = np.concatenate(observations)
        return observations
    
    def handle_global_states(self, global_states_dict):
        global_states_list = []
        global_states_list.append(global_states_dict['attack_remain_time']*0.05)
        global_states_list.append(global_states_dict['match_remain_time']*0.02)
        global_states_list.append(global_states_dict['is_home_team'])
        global_states_list.append(global_states_dict['ball_position_x']*0.2)
        global_states_list.append(global_states_dict['ball_position_y']*0.5)
        global_states_list.append(global_states_dict['ball_position_z']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_x']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_y']*0.5)
        global_states_list.append(global_states_dict['vec_ball_basket_z']*0.2)
        global_states_list.append(global_states_dict['team_own_ball'])
        global_states_list.append(global_states_dict['enemy_team_own_ball'])
        global_states_list.append(global_states_dict['ball_clear'])
        ball_status = onehot(int(global_states_dict['ball_status']), 6)
        global_states_list += ball_status
        global_states_list.append(global_states_dict['can_rebound'])
        global_states_list.append(global_states_dict['dis_to_rebound_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_y']*0.2)
        global_states_list.append(global_states_dict['can_block'])
        global_states_list.append(global_states_dict['shoot_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['shoot_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_y']*0.2)
        global_states_list.append(global_states_dict['block_diff_angle']*0.3)
        global_states_list.append(global_states_dict['block_diff_r']*0.2)
        return np.array(global_states_list)
    
    def handle_agent_states(self, agent_states_dict):
        agent_states_list = []
        agent_states_list.append(agent_states_dict['character_id']) # 文档有误，Jokic为3，Curry为2
        agent_states_list.append(agent_states_dict['position_type']) # 文档有误，1:C 2:PF 3:SF 4:SG 5:PG
        agent_states_list.append(agent_states_dict['buff_key'])
        agent_states_list.append(agent_states_dict['buff_value']*0.1)
        agent_states_list.append((agent_states_dict['stature']-180)*0.1)
        agent_states_list.append(agent_states_dict['rational_shoot_distance']-7)
        agent_states_list.append(agent_states_dict['position_x']*0.2)
        agent_states_list.append(agent_states_dict['position_y']*0.5)
        agent_states_list.append(agent_states_dict['position_z']*0.2)
        agent_states_list.append(agent_states_dict['v_delta_x']*0.3)
        agent_states_list.append(agent_states_dict['v_delta_z']*0.3)
        agent_states_list.append(agent_states_dict['player_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['player_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_me_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_me_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_basket_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_basket_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_ball_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_ball_r']*0.2)
        agent_states_list.append(agent_states_dict['facing_x'])
        agent_states_list.append(agent_states_dict['facing_y'])
        agent_states_list.append(agent_states_dict['facing_z'])
        agent_states_list.append(agent_states_dict['block_remain_best_time'])
        agent_states_list.append(agent_states_dict['block_remain_time'])
        agent_states_list.append(agent_states_dict['is_out_three_line'])
        agent_states_list.append(agent_states_dict['is_ball_owner'])
        agent_states_list.append(agent_states_dict['own_ball_duration']*0.2)
        agent_states_list.append(agent_states_dict['cast_duration'])
        agent_states_list.append(agent_states_dict['power']* 0.001)
        agent_states_list.append(agent_states_dict['is_cannot_dribble'])
        agent_states_list.append(agent_states_dict['is_pass_receiver'])
        agent_states_list.append(agent_states_dict['is_marking_opponent'])
        agent_states_list.append(agent_states_dict['is_team_own_ball'])
        agent_states_list.append(agent_states_dict['inside_defence'])
        is_my_team = onehot(int(agent_states_dict['is_my_team']), 2)
        agent_states_list += is_my_team
        player_state = onehot(int(agent_states_dict['player_state']), 6)
        agent_states_list += player_state
        skill_state = onehot(int(agent_states_dict['skill_state']), 27)
        agent_states_list += skill_state
        return np.array(agent_states_list)
    
    def rewards_wrapper(self, state_info):
        return self.handle_rewards(state_info[0], state_info[1])

    @staticmethod
    def rewards_default():
        return {
            "tot": 0,
            "state": 0,
            "state/not_ball_clear": 0,
            "state/attack_time_out": 0,
            # "state/free_ball": 0,
            "state/got_defended": 0,
            "state/out_of_defend": 0,
            "state/long_pass": 0,
            # "state/pass_fail": 0,
            # "state/no_defend_shoot": 0,
            "node": 0,
            "node/shoot": 0,
            "node/shoot/self_try": 0,
            "node/shoot/ally_try": 0,
            "node/shoot/enemy_try": 0,
            "node/shoot/inference": 0,
            "node/steal": 0,
            "node/steal/self_steal": 0,
            "node/steal/self_target_steal": 0,
            "node/block": 0,
            "node/block/self_block": 0,
            "node/block/ally_block": 0,
            "node/block/self_target_block": 0,
            "node/block/enemy_block": 0,
            "node/rebound": 0,
            "node/rebound/self_rebound": 0,
            "node/rebound/enemy_rebound": 0,
            "node/pickup": 0,
            "node/pickup/self_pickup": 0,
            "node/pickup/enemy_pickup": 0,
            "node/screen": 0,
            "dazhao": 0,
        }
    
    def handle_rewards(self, infos, states_dict):
        r = self.rewards_default()

        r["state"] = 0
        if infos.get("state_event",None) is not None:
            state_event_dict = infos["state_event"]
            if state_event_dict.get("not_ball_clear",None) is not None:
                r["state/not_ball_clear"] = - 0.0032
                r["state"] += r["state/not_ball_clear"]
            if state_event_dict.get("attack_time_out",None) is not None:
                r["state/attack_time_out"] = - 4
                r["state"] += r["state/attack_time_out"]
            # if state_event_dict.get("free_ball",None) is not None:
            #     # 0.0032 * 4 step/second * 20 second == 0.256
            #     r["state/free_ball"] = - 0.0032 * sqrt(states_dict['self_state']['ball_to_me_dis_x']**2 + states_dict['self_state']['ball_to_me_dis_z']**2) / 20 # changed
            #     r["state"] += r["state/free_ball"]
            if state_event_dict.get("got_defended",None) is not None:
                r["state/got_defended"] = - 0.2
                r["state"] += r["state/got_defended"]
            if state_event_dict.get("out_of_defend",None) is not None:
                r["state/out_of_defend"] = 0.2
                r["state"] += r["state/out_of_defend"]
            if state_event_dict.get("long_pass",None) is not None:
                r["state/long_pass"] = - 0.2
                r["state"] += r["state/long_pass"]
            # if state_event_dict.get("pass_fail",None) is not None:
            #     r["state/pass_fail"] = - 0.2 # changed
            #     r["state"] += r["state/pass_fail"]
            # if state_event_dict.get("no_defend_shoot",None) is not None:
            #     r["state/no_defend_shoot"] = 4 # changed
            #     r["state"] += r["state/no_defend_shoot"]

        r["node"] = 0
        if infos.get("shoot",None) is not None:
            shoot_dict = infos["shoot"]
            expect_score = (shoot_dict["two"] * 2 + shoot_dict["three"] * 3) * shoot_dict["hit_percent"]
            if shoot_dict["open_shoot"]:
                expect_score *= 1.2 # encourage open shoot
            r["node/shoot/self_try"] = expect_score * shoot_dict["me"]
            r["node/shoot/ally_try"] = expect_score * shoot_dict["ally"] * 0.5
            r["node/shoot/enemy_try"] = - expect_score * (shoot_dict["enemy"] * 0.8 + shoot_dict["opponent"] * 0.2)
            r["node/shoot/inference"] = shoot_dict["inference_degree"] * 0.5
            r["node/shoot"] = r["node/shoot/self_try"] + r["node/shoot/ally_try"] + r["node/shoot/enemy_try"] + r["node/shoot/inference"]
            r["node"] += r["node/shoot"]
        if infos.get("steal",None) is not None:
            steal_dict = infos["steal"]
            hit_percent = steal_dict["hit_percent"]
            r["node/steal/self_steal"] = hit_percent * steal_dict["me"] * 0.5
            r["node/steal/self_target_steal"] = - hit_percent * steal_dict["target"] * 0.5
            r["node/steal"] = r["node/steal/self_steal"] + r["node/steal/self_target_steal"]
            r["node"] += r["node/steal"]
        if infos.get("block",None) is not None:
            block_dict = infos["block"]
            expected_score = block_dict["expected_score"]
            hit_percent = block_dict["hit_percent"]
            r["node/block/self_block"] = expected_score * hit_percent * block_dict["me"]
            r["node/block/ally_block"] = expected_score * hit_percent * block_dict["ally"] * 0.8
            r["node/block/self_target_block"] = - expected_score * hit_percent * block_dict["target"]
            r["node/block/enemy_block"] = - expected_score * hit_percent * block_dict["enemy"] * 0.5
            r["node/block"] = r["node/block/self_block"] + r["node/block/ally_block"] + r["node/block/self_target_block"] + r["node/block/enemy_block"]
            r["node"] += r["node/block"]
        if infos.get("rebound",None) is not None:
            rebound_dict = infos["rebound"]
            r["node/rebound/self_rebound"] = rebound_dict["me"] * 0.3
            r["node/rebound/enemy_rebound"] = - rebound_dict["enemy"] * 0.3
            r["node/rebound"] = r["node/rebound/self_rebound"] + r["node/rebound/enemy_rebound"]
            r["node"] += r["node/rebound"]
        if infos.get("pickup",None) is not None:
            pickup_dict = infos["pickup"]
            r["node/pickup/self_pickup"] = pickup_dict["me"] * pickup_dict["success"] * 0.3
            r["node/pickup/enemy_pickup"] = - pickup_dict["enemy"] * pickup_dict["success"] * 0.3
            r["node/pickup"] = r["node/pickup/self_pickup"] + r["node/pickup/enemy_pickup"]
            r["node"] += r["node/pickup"]
        if infos.get("screen",None) is not None:
            screen_dict = infos["screen"]
            r["node/screen"] = 0
            if screen_dict["position"] < 3:
                r["node/screen"] += screen_dict["me"] * 0.2
            r["node"] += r["node/screen"]

        r["dazhao"] = 0
        if len(infos["end_values"]) > 0:
            end_values_dict = infos["end_values"]
            r["dazhao"] += end_values_dict["my_dazhao_cnt"] * 0.2

        r["tot"] = r["state"] + r["node"] + r["dazhao"]

        return r


class BCWrapper():
    ''' Simple Wrapper for Baseline
    '''
    def __init__(self, config):
        self.config = config

    def state_wrapper(self, states_dict):
        global_state = self.handle_global_states(states_dict['global_states']) # 
        self_state = self.handle_agent_states(states_dict['self_state'])
        ally0_state = self.handle_agent_states(states_dict['ally_0_state'])
        ally1_state = self.handle_agent_states(states_dict['ally_1_state'])
        enemy0_state = self.handle_agent_states(states_dict['enemy_0_state'])
        enemy1_state = self.handle_agent_states(states_dict['enemy_1_state'])
        enemy2_state = self.handle_agent_states(states_dict['enemy_2_state'])
        states = [global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state]
        return states

    def handle_global_states(self, global_states_dict):
        global_states_list = []
        global_states_list.append(global_states_dict['attack_remain_time']*0.05)
        global_states_list.append(global_states_dict['match_remain_time']*0.02)
        global_states_list.append(global_states_dict['is_home_team'])
        global_states_list.append(global_states_dict['ball_position_x']*0.2)
        global_states_list.append(global_states_dict['ball_position_y']*0.5)
        global_states_list.append(global_states_dict['ball_position_z']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_x']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_y']*0.5)
        global_states_list.append(global_states_dict['vec_ball_basket_z']*0.2)
        global_states_list.append(global_states_dict['team_own_ball'])
        global_states_list.append(global_states_dict['enemy_team_own_ball'])
        global_states_list.append(global_states_dict['ball_clear'])
        ball_status = onehot(int(global_states_dict['ball_status']), 6)
        global_states_list += ball_status
        global_states_list.append(global_states_dict['can_rebound'])
        global_states_list.append(global_states_dict['dis_to_rebound_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_y']*0.2)
        global_states_list.append(global_states_dict['can_block'])
        global_states_list.append(global_states_dict['shoot_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['shoot_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_y']*0.2)
        global_states_list.append(global_states_dict['block_diff_angle']*0.3)
        global_states_list.append(global_states_dict['block_diff_r']*0.2)
        return np.array(global_states_list)
    def handle_agent_states(self, agent_states_dict):
        agent_states_list = []
        agent_states_list.append(agent_states_dict['character_id'])
        agent_states_list.append(agent_states_dict['position_type'])
        # position_type = onehot(int(agent_states_dict['position_type']) - 1, 5)
        # agent_states_list += position_type
        agent_states_list.append(agent_states_dict['buff_key'])
        agent_states_list.append(agent_states_dict['buff_value']*0.1)
        agent_states_list.append((agent_states_dict['stature']-180)*0.1)
        agent_states_list.append(agent_states_dict['rational_shoot_distance']-7)
        agent_states_list.append(agent_states_dict['position_x']*0.2)
        agent_states_list.append(agent_states_dict['position_y']*0.5)
        agent_states_list.append(agent_states_dict['position_z']*0.2)
        agent_states_list.append(agent_states_dict['v_delta_x']*0.3)
        agent_states_list.append(agent_states_dict['v_delta_z']*0.3)
        agent_states_list.append(agent_states_dict['player_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['player_to_me_dis_z']*0.2) # z
        agent_states_list.append(agent_states_dict['basket_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_z']*0.2)# z
        agent_states_list.append(agent_states_dict['ball_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_z']*0.2)# z
        agent_states_list.append(agent_states_dict['polar_to_me_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_me_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_basket_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_basket_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_ball_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_ball_r']*0.2)
        agent_states_list.append(agent_states_dict['facing_x'])
        agent_states_list.append(agent_states_dict['facing_y'])
        agent_states_list.append(agent_states_dict['facing_z'])
        agent_states_list.append(agent_states_dict['block_remain_best_time'])
        agent_states_list.append(agent_states_dict['block_remain_time'])
        agent_states_list.append(agent_states_dict['is_out_three_line'])
        agent_states_list.append(agent_states_dict['is_ball_owner'])
        agent_states_list.append(agent_states_dict['own_ball_duration']*0.2)
        agent_states_list.append(agent_states_dict['cast_duration'])
        agent_states_list.append(agent_states_dict['power']*0.001)
        agent_states_list.append(agent_states_dict['is_cannot_dribble'])
        agent_states_list.append(agent_states_dict['is_pass_receiver'])
        agent_states_list.append(agent_states_dict['is_marking_opponent'])
        agent_states_list.append(agent_states_dict['is_team_own_ball'])
        agent_states_list.append(agent_states_dict['inside_defence'])
        is_my_team = onehot(int(agent_states_dict['is_my_team']), 2)
        agent_states_list += is_my_team
        player_state = onehot(int(agent_states_dict['player_state']), 6)
        agent_states_list += player_state
        skill_state = onehot(int(agent_states_dict['skill_state']), 27)
        agent_states_list += skill_state
        return np.array(agent_states_list)
    