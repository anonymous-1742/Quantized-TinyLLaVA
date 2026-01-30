from packaging import version
import pathlib
import sys
sys.path.append("/scratch/drjieliu_root/drjieliu99/gjiajun/TinyLLaVA_Factory")
import tokenizers
import transformers


from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')



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

class NumpyArrayWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'wb')
        
    def write_array(self, array):
        # 确保是numpy数组，并转换为float16
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        array = array.astype(np.float16)
        
        # 写入维数
        dim = array.ndim
        self.file.write(struct.pack('<i', dim))
        
        # 写入形状
        for s in array.shape:
            self.file.write(struct.pack('<i', s))
            
        # 写入dtype名称
        dtype_name = array.dtype.name.encode('utf-8')
        self.file.write(struct.pack('<i', len(dtype_name)))
        self.file.write(dtype_name)
        
        # 写入数组数据
        self.file.write(array.tobytes())
        
    def close(self):
        self.file.close()
        
def train():
    # load argument
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
    model_server =TinyLlavaSplitServer(config=model_config)
    VQ_trainer_argument=VQ_train_config(0.1,0.3,0.6)
    print("model generated")
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path !="" and training_arguments.pretrained_model_path is not None:
        model_client = training_recipe.load(model_client, model_args)
        model_server = training_recipe.load(model_server, model_args)
    else:
        model_client.load_llm(**model_args['llm'])
        model_client.load_vision_tower(**model_args['vision_tower'])
        model_client.load_connector(**model_args['connector'])
        model_server.load_llm(**model_args['llm'])
    model_client = training_recipe(model_client)
    model_server = training_recipe(model_server)
    #model.split_model(model_config)
    model_client.config.use_cache = False
    model_client.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model_client.tokenizer
    model_server.config.use_cache = False
    model_server.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model_client.tokenizer
    data_arguments.image_processor = model_client.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_arguments)
    print("data_loaded")
    full_train_dataset = data_module['train_dataset']
    subset_size = int(0.1 * len(full_train_dataset))  # 10% 的子集
    indices = list(range(subset_size))
    #data_module['train_dataset'] = torch.utils.data.Subset(full_train_dataset, indices)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    trainer.train()
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
