import queue
import time
import socketserver

import baselines.rlsdk as rlsdk
import traceback

def is_end_state(state):
    infos = state[0]
    if infos.get('end_values', None) is not None and len(infos['end_values']) > 0:
        return True
    return False

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """TCP Service Handler for Dunk City Dynasty game client 

    [V1.0] Multi-thread async TCP handler

    Args:
        request: default setting
        client_address: default setting
        server: default setting
        data: stream data for state & action of reinforcement learning

    Attributes:
        [handle]: data transmission function
    """

    def __init__(self, request, client_address, server, stream_data, tprint) -> None:
        self.stream_data = stream_data
        self.tprint = tprint
        super().__init__(request, client_address, server)

    def setup(self):
        """server setup
        """
        self.request.setsockopt(socketserver.socket.IPPROTO_TCP, socketserver.socket.TCP_NODELAY, True)

    def handle(self):
        """data transmission function
        """
        try:
            self.stream_data['established'].acquire(timeout=1)
            try:
                self.tprint("comm: conn established")
                self.data = bytes()
                while True:
                    self.data += self.request.recv(10240000)
                    while True:
                        head_length, msg = rlsdk.unpack_photon_rpc_head(self.data) # 这里也可能会block（一个进程里可能有很多thread卡在这里），但是修不了，因为不是我实现的
                        if head_length == 0:
                            break
                        self.data = self.data[head_length:]

                        msg_type, result = rlsdk.RlsdkDeserializer.deserialize(msg)
                        if msg_type == rlsdk.RLSDKMsgType.STARTINFO:
                            response = rlsdk.RlsdkDeserializer.serialize_recv_start_info(result.transaction_id, True, '\"\"')
                            response_with_head = rlsdk.pack_photon_h(len(response)) + response
                            time.sleep(0.01)
                            self.request.sendall(response_with_head)
        
                        elif msg_type == rlsdk.RLSDKMsgType.STATES:
                            team_id, member_id = result.agent_id.team_id, result.agent_id.member_id
                            agent_id = team_id * 3 + member_id
    
                            assert result.state

                            start_time = time.time()

                            self.stream_data['state'].put((agent_id, result.state), timeout=10) # block # TODO 验证
                                
                            agent_id_, action = self.stream_data['action'].get(timeout=1) # may timeout when game done
                            
                            assert agent_id == agent_id_

                            # set minimum game execution time (5ms)
                            sleep_time = 0.005 - (time.time() - start_time)
                            if sleep_time > 0:
                                time.sleep(sleep_time)

                            if time.time() - start_time > 0.050:
                                self.tprint(f"comm: step time > 50ms {time.time() - start_time}")

                            response = rlsdk.RlsdkDeserializer.serialize_action(result.transaction_id, [action])
                            response_with_head = rlsdk.pack_photon_h(len(response)) + response
                            self.request.sendall(response_with_head)
            finally:
                self.stream_data['established'].release()
                self.tprint("comm: conn released")
        except queue.Empty:
            self.tprint(f"comm: game done")
        except Exception as e:
            self.tprint(f"comm: ignored exception @handle: {traceback.format_exc()}")

        
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """TCP Service for  Dunk City Dynasty game client 

    [V1.0] Shared memory version

    Attributes:
        [finish_request]: override default finish_request func with shared memory and thread lock
    """
    def finish_request(self, request, client_address):
        """Finish one request by instantiating RequestHandlerClass
        """
        self.rhc = self.RequestHandlerClass(request, client_address, self, self.stream_data, self.tprint)
        