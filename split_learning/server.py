from tinyllava.model.VQ.vq import FSQ_block,VQ_config
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import quantize as qt
import socket
import struct
import pickle

class Server(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear2=nn.Sequential(nn.Linear(100,3),nn.Softmax())
        config=VQ_config(token_dim=100,code_dim=100,discrete_size=4)
        self.VQ=FSQ_block(config)
        self.quantizer=self.VQ.quantizer
    def forward(self,payload,aux,quantize=False):
        if quantize:
            payload=self.quantizer.decompress(payload,aux)
        x=self.Linear2(payload)
        return(x)
    
class DummyDataset(Dataset):
    def __init__(self, num_samples=200, seq_len=100, num_classes=3):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.labels = torch.randint(0, num_classes, (num_samples,seq_len,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.labels[idx] 

def send_tensor(sock, tensor: torch.Tensor):
    """
    Send a torch Tensor through a socket
    """
    try:
        # 1. detach + cpu
        if isinstance(tensor, torch.Tensor):
            tensor_cpu = tensor.detach().cpu()

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
        print(f"✓ Deserialized tensor")

        return tensor

    except Exception as e:
        print(f"✗ receive_tensor error: {e}")
        raise
        
import socket
import traceback
import torch
import torch.nn as nn

def safe_close(sock, name):
    try:
        if sock:
            sock.close()
            print(f"[INFO] {name} closed")
    except Exception as e:
        print(f"[WARN] failed to close {name}: {e}")

fwd_server = bwd_server = fwd_sock = bwd_sock = None

try:
    model = Server()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    host = '0.0.0.0'
    forward_port = 5001
    backward_port = 5002

    # ===== forward socket =====
    fwd_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fwd_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    fwd_server.bind((host, forward_port))
    fwd_server.listen(1)
    print("[INFO] waiting for forward connection...")
    fwd_sock, _ = fwd_server.accept()
    print("[INFO] forward connected")

    # ===== backward socket =====
    bwd_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bwd_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bwd_server.bind((host, backward_port))
    bwd_server.listen(1)
    print("[INFO] waiting for backward connection...")
    bwd_sock, _ = bwd_server.accept()
    print("[INFO] backward connected")

    trainset = DummyDataset()
    trainloader = DataLoader(trainset, batch_size=8)

    quantize = True

    for step, target in enumerate(trainloader):
        try:
            optimizer.zero_grad()

            data_pack = receive_tensor(fwd_sock)

            if quantize:
                packed_payload, pad, aux, pack_shape = data_pack
                payload = qt.unpack_2bit_tensor(packed_payload, pad)
                payload = payload.reshape(pack_shape)
            else:
                payload = data_pack
                aux = None

            payload = payload.to(next(model.parameters()).device)
            payload.requires_grad_(True)

            # ===== forward =====
            pred = model(payload, aux, quantize)
            loss = loss_fn(pred, target)
            print(f"[INFO][step {step}] loss = {loss.item():.4f}")

            # ===== backward =====
            loss.backward()
            optimizer.step()

            # ===== send gradient =====
            send_tensor(bwd_sock, payload.grad)

        except Exception:
            print(f"[ERROR] exception at training step {step}")
            traceback.print_exc()
            break

except Exception:
    print("[FATAL] server crashed during initialization")
    traceback.print_exc()

finally:
    print("[INFO] shutting down server...")
    safe_close(fwd_sock, "fwd_sock")
    safe_close(bwd_sock, "bwd_sock")
    safe_close(fwd_server, "fwd_server")
    safe_close(bwd_server, "bwd_server")
    print("[INFO] server shutdown complete")


