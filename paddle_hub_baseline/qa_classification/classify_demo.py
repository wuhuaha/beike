# 句对分类
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

class TextClassification(BaseNLPDataset):
    # 句对分类数据集
    # 通过传入不同数据集文件夹，来使用不同数据集
    '''
        bq_corpus = TextClassification('data/bq_corpus')
        paws-x-zh = TextClassification('data/paws-x-zh')
        lcqmc = TextClassification('data/lcqmc')
    '''
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        super(TextClassification, self).__init__(
            base_path=self.dataset_dir,
            train_file="classify_train.tsv",
            dev_file="classify_dev.tsv",
            predict_file="classify_test.tsv",
            train_file_with_header=False,
            dev_file_with_header=False,
            predict_file_with_header=False,
            label_list=["0", "1"])

# 数据集设置，通过更换数据集文件夹更换数据集
dataset_dir = './bq_corpus'
dataset_name = dataset_dir.split('/')[-1]

# 加载语义模型，可更换其他语义模型比如bert、robert等
module = hub.Module(name="ernie")
inputs, outputs, program = module.context(
        trainable=True, max_seq_len=256)
metrics_choices = ["acc"]

pooled_output = outputs["pooled_output"]

# 定义数据集
dataset = TextClassification(dataset_dir)

# 定义数据读取器
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=128,
    do_lower_case=True,
    sp_model_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path()
)

# 设置优化策略
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,
    lr_scheduler="linear_decay",
    warmup_proportion=0.1,
    weight_decay=0.01,
    optimizer_name="adam"
)

# 设置训练参数
config = hub.RunConfig(
    log_interval=20,
    eval_interval=500,
    use_pyreader=True,
    use_data_parallel=True,
    save_ckpt_interval=1000,
    use_cuda=True,
    checkpoint_dir="%s_TextClassification" % dataset_name,
    num_epoch=5,
    batch_size=128,
    strategy=strategy)

# 模型上下文设置
inputs, outputs, program = module.context(
    trainable=True,
    max_seq_len=400
)

# 语义模型输出
sequence_output = outputs["sequence_output"]


# 模型输入
feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

# 训练任务
cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=metrics_choices
        )
# 开始训练
run_states = cls_task.finetune_and_eval()
print(cls_task.predict(data=dataset))
