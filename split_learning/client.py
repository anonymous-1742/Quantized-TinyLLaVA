from tinyllava.model.VQ.vq import FSQ_block,VQ_config
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import socket
import struct
import pickle
import quantize as qt
import traceback

class Client(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1=nn.Sequential(nn.Linear(100,100),nn.ReLU())
        config=VQ_config(token_dim=100,code_dim=100,discrete_size=4)
        self.VQ=FSQ_block(config)
        self.quantizer=self.VQ.quantizer
        
    def forward(self,x,quantize=False):
        x=self.Linear1(x)
        #x,L_comm=self.VQ(x,return_indice=True)
        
        if quantize:
            payload,aux,vq_loss=self.quantizer.compress(x)
            return(x,payload,aux,vq_loss)
        else:
            return(x,0,0,0)
    
class DummyDataset(Dataset):
    def __init__(self, num_samples=200,seq_len=100 ,num_features=100, num_classes=3):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.data = torch.randn(num_samples,seq_len, num_features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def send_tensor(sock, tensor: torch.Tensor):
    """
    Send a torch Tensor through a socket
    """
    try:
        # 1. detach + cpu
        if isinstance(tensor, torch.Tensor):
            tensor_cpu = tensor.detach().cpu
        else:
            tensor_cpu = tensor

        # 2. serialize
        data = pickle.dumps(tensor_cpu)
        data_length = len(data)

        print(f"✓ Serialized tensor: {data_length} bytes")

        # 3. send length
        sock.sendall(data_length.to_bytes(4, byteorder='big'))
        print(f"✓ Sent length: {data_length} bytes")

        # 4. send payload
        sock.sendall(data)

    except Exception as e:
        print(f"✗ send_tensor error: {e}")
        raise
    
def receive_tensor(sock):
    """
    Receive a torch Tensor from socket
    """
    try:
        # 1. receive length
        length_bytes = sock.recv(4)
        if not length_bytes:
            return None

        data_length = int.from_bytes(length_bytes, byteorder='big')
        print(f"✓ Expecting {data_length} bytes")

        # 2. receive payload
        received_data = b''
        while len(received_data) < data_length:
            chunk = sock.recv(min(4096, data_length - len(received_data)))
            if not chunk:
                raise RuntimeError("Socket connection broken")
            received_data += chunk

        print(f"✓ Received {len(received_data)} bytes")

        # 3. deserialize
        tensor = pickle.loads(received_data)
        print(f"✓ Deserialized tensor, shape={tensor.shape}")

        return tensor

    except Exception as e:
        print(f"✗ receive_tensor error: {e}")
        raise

def safe_close(sock, name):
    try:
        if sock:
            sock.close()
            print(f"[INFO] {name} closed")
    except Exception as e:
        print(f"[WARN] failed to close {name}: {e}")

fwd_sock = bwd_sock = None

try:
    model = Client()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    server_host = 'localhost'
    forward_port = 5001
    backward_port = 5002

    quantize = True

    trainset = DummyDataset()
    trainloader = DataLoader(trainset, batch_size=8)

    # ===== connect =====
    fwd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fwd_sock.connect((server_host, forward_port))
    print("[INFO] connected to server (forward)")

    bwd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bwd_sock.connect((server_host, backward_port))
    print("[INFO] connected to server (backward)")

    for step, data in enumerate(trainloader):
        try:
            optimizer.zero_grad()

            # ===== forward =====
            if quantize:
                x, payload, aux, vq_loss = model(data, quantize)
                packed_payload, pad = qt.pack_2bit_tensor(payload.to(torch.uint8))
                pack_shape = payload.shape
                data_pack = packed_payload, pad, aux, pack_shape
            else:
                x, _ = model(data, quantize)
                data_pack = x

            send_tensor(fwd_sock, data_pack)

            x.requires_grad_(True)

            # ===== receive gradient =====
            grad = receive_tensor(bwd_sock)

            # ===== backward =====
            x.backward(grad)
            optimizer.step()

            print(f"[INFO][step {step}] client step done")

        except Exception:
            print(f"[ERROR] exception at client step {step}")
            traceback.print_exc()
            break

except Exception:
    print("[FATAL] client crashed during initialization")
    traceback.print_exc()

finally:
    print("[INFO] shutting down client...")
    safe_close(fwd_sock, "fwd_sock")
    safe_close(bwd_sock, "bwd_sock")
    print("[INFO] client shutdown complete")
