import socket
import pickle
import struct
import time

'''
def send_obj(sock, obj):
    sent_time=time.time()
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)
'''
def send_obj(sock, obj):
    sent_time = time.time()

    data = pickle.dumps(obj)
    payload_bytes = len(data)
    header_bytes = 4
    total_bytes = header_bytes + payload_bytes

    # 发送长度头
    sock.sendall(struct.pack(">I", payload_bytes))
    # 发送数据
    sock.sendall(data)


    return sent_time


'''
def recv_obj(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    length = struct.unpack(">I", raw_len)[0]
    data = recvall(sock, length)
    rec_time=time.time()
    return pickle.loads(data),rec_time
'''
def recv_obj(sock):
    # 接收长度头
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None

    header_bytes = 4
    payload_len = struct.unpack(">I", raw_len)[0]

    # 接收 payload
    data = recvall(sock, payload_len)
    rec_time = time.time()

    total_bytes = header_bytes + payload_len
    total_mb=total_bytes/1000/1024
    return pickle.loads(data), rec_time, total_mb

def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet

    return data