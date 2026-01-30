from packaging import version
import pathlib
import sys
import tokenizers
import transformers


from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module
import socket
import time
from torch.utils.data import DataLoader
from comm import send_obj, recv_obj
import traceback
import os


def save_client_model(model, output_dir):
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

    print("[Client] Model saved successfully")


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
device=torch.device("cuda")
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
model_client =TinyLlavaSplitClient(config=model_config)
VQ_trainer_argument=VQ_train_config(0.1,0.3,0.6)
print("model generated")
    # load pretrained checkpoint
if training_arguments.pretrained_model_path !="" and training_arguments.pretrained_model_path is not None:
        model_client = training_recipe.load(model_client, model_args)
else:
        model_client.load_llm(**model_args['llm'])
        model_client.load_vision_tower(**model_args['vision_tower'])
        model_client.load_connector(**model_args['connector'])
model_client = training_recipe(model_client)
model_client.config.use_cache = False
model_client.config.image_aspect_ratio = data_arguments.image_aspect_ratio
tokenizer = model_client.tokenizer
tokenizer = model_client.tokenizer
data_arguments.image_processor = model_client.vision_tower._image_processor
data_arguments.is_multimodal = True
data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_arguments)
print("data_loaded")
#optimizer_client = torch.optim.Adam(model_client.parameters(), lr=1e-3)

trainer_client = LLaVATrainer(model=model_client, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)

optimizer_client=trainer_client.create_optimizer()

sampler=trainer_client._get_train_sampler()


train_loader = DataLoader(
    data_module['train_dataset'],
    batch_size=training_arguments.per_device_train_batch_size,
    collate_fn=data_module['data_collator'],
    sampler=sampler
)


import socket
import time
import torch
from comm import send_obj, recv_obj
from tqdm import tqdm  # 新增 tqdm

HOST = "10.224.37.213"
PORT = int(os.environ.get("PORT", 1111))
print(f"[Client] Connecting to {HOST}:{PORT}")
device = model_client.device

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print("[Client] Connected to server")

    max_step = 100
    # 使用 tqdm 包裹 train_loader，显示进度条
    for step, data in enumerate(tqdm(train_loader, desc="[Client] Training Progress")):
        t0 = time.time()

        input_ids = data['input_ids'].to(device)
        labels = data['labels'].to(device)
        attention_mask = data['attention_mask'].to(device)
        images = data['images'].to(device)

        # ===== Client forward =====
        transmission, vq_loss, image_embeds_c, text_embeds_c = model_client(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=labels
        )    

        # ===== 发送 =====
        sent_time = time.time()
        transmission["sent_time"] = sent_time
        print("[Client] sending transmission")
        send_obj(sock, transmission)
        
        # ===== 接收梯度 =====
        response, rec_time,_ = recv_obj(sock)
        if response is None:
            print("[Client] Server closed connection")
            break

        img_grad = response["img_grad"].to(device)
        txt_grad = response["txt_grad"].to(device)

        # ===== Client backward =====
        optimizer_client.zero_grad()
        image_embeds_c.backward(img_grad, retain_graph=True)
        text_embeds_c.backward(txt_grad, retain_graph=True)
        if vq_loss is not None and isinstance(vq_loss, torch.Tensor):
            vq_loss_record = vq_loss.item()
            vq_loss.backward()
        else:
            vq_loss_record = 0 
        optimizer_client.step()

        # ===== 更新 tqdm 描述，显示当前 VQ loss =====
        tqdm.write(f"[Client] Step {step} | VQ loss: {vq_loss_record:.4f}")

except KeyboardInterrupt:
    print("\n[Client] Interrupted by user")

except Exception as e:
    print(f"[Client] Error: {e}")
    traceback.print_exc() 

finally:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    save_client_model(model_client, training_arguments.output_dir)
    sock.close()
    print("[Client] Socket closed")