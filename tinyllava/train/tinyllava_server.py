from packaging import version
import pathlib
import sys
sys.path.append("/scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory")
import tokenizers
import transformers
import traceback
import os
from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
device=torch.device("cuda")

import torch
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

def save_server_model(model, output_dir):
    """
    等价于你给的 save() 中的 pretrain 阶段保存逻辑
    （不依赖 Trainer / DeepSpeed）
    """
    print("[Client] Saving model...")

    # 1. save config
    if hasattr(model, "config"):
        model.config.use_cache = True
        model.config.save_pretrained(output_dir)

    # 2. save tokenizer（如果存在）
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        model.tokenizer.save_pretrained(output_dir)

    # 3. save language model
    if hasattr(model, "language_model"):
        lm_dir = os.path.join(output_dir, "language_model")
        os.makedirs(lm_dir, exist_ok=True)
        torch.save(
            model.language_model.state_dict(),
            os.path.join(lm_dir, "pytorch_model.bin")
        )
        if hasattr(model.config, "text_config"):
            model.config.text_config.save_pretrained(lm_dir)

    # 4. save vision tower
    if hasattr(model, "vision_tower"):
        vt = model.vision_tower._vision_tower
        vt_dir = os.path.join(output_dir, "vision_tower")
        os.makedirs(vt_dir, exist_ok=True)
        torch.save(
            vt.state_dict(),
            os.path.join(vt_dir, "pytorch_model.bin")
        )
        if hasattr(vt, "config"):
            vt.config.save_pretrained(vt_dir)

    # 5. save connector
    if hasattr(model, "connector"):
        conn_dir = os.path.join(output_dir, "connector")
        os.makedirs(conn_dir, exist_ok=True)
        torch.save(
            model.connector.state_dict(),
            os.path.join(conn_dir, "pytorch_model.bin")
        )

    # 6. save VQ-VAE
    if hasattr(model, "VQ"):
        vq_dir = os.path.join(output_dir, "VQ_VAE")
        os.makedirs(vq_dir, exist_ok=True)
        torch.save(
            model.VQ.state_dict(),
            os.path.join(vq_dir, "pytorch_model.bin")
        )

    print("[Server] Model saved successfully")

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments)
    model_args['vq'] = _load_vq_settings(model_arguments)
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vq_settings(model_arguments):
    vq_args={}
    vq_args['vq_type']=model_arguments.vq_type
    return vq_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args

class VQ_train_config:
    def __init__(self,label_smoothing_factor,comm_weight,code_weight):
        self.comm_weight=comm_weight
        self.code_weight=code_weight
        self.label_smoothing_factor=label_smoothing_factor
        
parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
logger_setting(getattr(training_arguments, 'output_dir', None))
training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
model_args = load_settings(model_arguments, data_arguments, training_arguments)
model_args = training_recipe.add_args(model_args)
model_config = TinyLlavaConfig()
model_config.load_from_config(model_arguments)
model_server =TinyLlavaSplitServer(config=model_config)
VQ_trainer_argument=VQ_train_config(0.1,0.3,0.6)
print("model generated")
    # load pretrained checkpoint
if training_arguments.pretrained_model_path !="" and training_arguments.pretrained_model_path is not None:
        model_server = training_recipe.load(model_server, model_args)
else:
        model_server.load_llm(**model_args['llm'])
model_server = training_recipe(model_server)
#model.split_model(model_config)

model_server.config.use_cache = False
model_server.config.image_aspect_ratio = data_arguments.image_aspect_ratio
tokenizer = model_server.tokenizer
#data_module['train_dataset'] = torch.utils.data.Subset(full_train_dataset, indices)
#log_trainable_params(model_client)  # not work well with zero3
#log_trainable_params(model_server) 
#data_module['train_dataset'] = torch.utils.data.Subset(full_train_dataset, indices)
trainer_server = LLaVATrainer(model=model_server, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments)
print("trainer:",trainer_server)

import socket
import time
import torch
from comm import send_obj, recv_obj

HOST = "10.224.37.213"
PORT = int(os.environ.get("PORT", 1111))
print(f"[Client] Connecting to {HOST}:{PORT}")
device = model_server.device
optimizer_server=trainer_server.create_optimizer()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# ⭐ 关键：允许端口复用
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

conn = None

import logging
import traceback
import socket

# ===== 日志设置 =====
logging.basicConfig(
    filename="server_latency.log",   # 保存的文件名
    filemode="w",                    # 每次运行覆盖旧日志；改成 "a" 可追加
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def obj_bits(obj):
    """
    Recursively compute bits of all tensors inside obj
    """
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size() * 8

    elif isinstance(obj, (list, tuple)):
        return sum(obj_bits(o) for o in obj)

    elif isinstance(obj, dict):
        return sum(obj_bits(v) for v in obj.values())

    elif isinstance(obj, (int, float)):
        # Python 标量：按 float64 / int64 计算（最保守）
        return 64

    else:
        # 其他类型（None / str / bool 等）
        return 0

try:
    sock.bind((HOST, PORT))
    sock.listen(1)
    latencies_per_100steps=[]
    print("[Server] Waiting for client...")
    logging.info("[Server] Waiting for client...")

    conn, addr = sock.accept()
    print(f"[Server] Connected from {addr}")
    logging.info(f"[Server] Connected from {addr}")

    step = 0
    latencies = []
    cumulative_latency = 0.0  # 用于每100步累计
    mb_num=[]
    while True:
        received= recv_obj(conn)
        print("[Server] received transmission")

        if received is None:
            print("[Server] Client disconnected")
            logging.info("[Server] Client disconnected")
            break

        # ===== 解包 =====
        transmission, rec_time, total_mb= received
        print(f"[Server] received {total_mb} MB")
        mb_num.append(total_mb)
        image_payloads = transmission["image_payloads"]
        image_aux = transmission["image_aux"]
        text_payloads = transmission["text_payloads"]
        text_aux = transmission["text_aux"]
        attention_mask = transmission["attention_mask"].to(device)
        labels = transmission["labels"].to(device)
        position_ids = transmission["position_ids"]
        text_positions = transmission["text_positions"]
        sent_time = transmission["sent_time"]
        uplink_bits = obj_bits(transmission)
        # ===== 解压 =====
        image_embeds_s, text_embeds_s = model_server.decompress_embeds(
            image_payloads, image_aux, text_payloads, text_aux
        )
        image_embeds_s = image_embeds_s.detach().requires_grad_(True)
        text_embeds_s = text_embeds_s.detach().requires_grad_(True)

        server_input_embeds = model_server.concat_embeds(
            image_embeds_s, text_embeds_s, text_positions
        )

        # ===== 前向 =====
        output = model_server.forward(
            server_input_embeds,
            attention_mask,
            position_ids,
            labels
        )

        loss = output["loss"]

        optimizer_server.zero_grad()
        loss.backward()

        img_grad = image_embeds_s.grad.detach()
        txt_grad = text_embeds_s.grad.detach()

        optimizer_server.step()

        # ===== 发送梯度 =====
        send_obj(conn, {
            "img_grad": img_grad,
            "txt_grad": txt_grad
        })

        # ===== latency 计算 =====
        latency = rec_time - sent_time
        latencies.append(latency)
        cumulative_latency += latency
        step += 1

        # ===== 每步打印 & 写入日志 =====
        step_msg = f"[Server] Step {step} | Loss: {loss.item():.4f} | Latency: {latency:.3f}s"
        print(step_msg)
        logging.info(step_msg)
        # ===== 每100步打印 & 写入平均 latency =====
        if step % 100 == 0:
            latencies_per_100steps.append(cumulative_latency)
            cum_msg = f"[Server] Step {step} | Total latency over last 100 steps: {cumulative_latency:.3f}s"
            print(cum_msg)
            logging.info(cum_msg)
            cumulative_latency = 0.0  # 重置累计

except KeyboardInterrupt:
    print("\n[Server] Interrupted by user")
    logging.info("[Server] Interrupted by user")

except Exception as e:
    print(f"[Server] Error: {e}")
    logging.error(f"[Server] Error: {e}")
    logging.error(traceback.format_exc())

finally:
    if conn is not None:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        conn.close()
        print("[Server] Connection closed")
        logging.info("[Server] Connection closed")

    sock.close()
    print("[Server] Socket closed")
    logging.info("[Server] Socket closed")

    save_server_model(model_server, training_arguments.output_dir)
    logging.info("[Server] Model saved.")

    if latencies:
        avg_latency = sum(latencies_per_100steps) / len(latencies_per_100steps)
        avg_msg = f"[Server] Average latency per 100 steps: {avg_latency:.3f}s"
        avg_bytes_msg= f"[Server] Average MB transmission per steps: {sum(mb_num)/len(mb_num):.3f} MB"
        total_bytes_msg= f"[Server] Total MB transmission: {sum(mb_num):.3f} MB"
        total_time_msg = f"[Server] Total latency: {sum(latencies):.3f}s"
        print(avg_msg)
        print(total_time_msg)
        print(avg_bytes_msg)
        print(total_bytes_msg)
        logging.info(avg_msg)