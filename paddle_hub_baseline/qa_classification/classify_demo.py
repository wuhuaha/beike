# 句对分类
import argparse
import ast
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
import csv
import io
import numpy as np
import pandas as pd

#读取test数据集 id id
def GetFileRecord(input_file):
  with io.open(input_file, "r", encoding="UTF-8") as file:
    reader = csv.reader(file, delimiter="\t", quotechar=None)
    records = list()
    ids = list()
    for (i, line) in enumerate(reader):
      id = [line[0], line[1]]
      record = [line[2], line[3]]
      records.append(record)
      ids.append(id)
  return (records,ids)

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

class TextClassification(BaseNLPDataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        super(TextClassification, self).__init__(
            base_path=self.dataset_dir,
            train_file="classify_train.tsv",
            dev_file="classify_dev.tsv",
            train_file_with_header=False,
            dev_file_with_header=False,
            label_list=["0", "1"])

# 数据集设置，通过更换数据集文件夹更换数据集
dataset_dir = './classify'
dataset_name = dataset_dir.split('/')[-1]

# 加载语义模型，可更换其他语义模型比如bert、robert等
module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)
metrics_choices = ["acc"]

# Construct transfer learning network
# Use "pooled_output" for classification tasks on an entire sentence.
# Use "sequence_output" for token-level output.
pooled_output = outputs["pooled_output"]

# 定义数据集
dataset = TextClassification(dataset_dir)

# 定义数据读取器
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=args.max_seq_len,
    do_lower_case=True,
    sp_model_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path()
)

# 设置优化策略
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=args.learning_rate,
    lr_scheduler="linear_decay",
    warmup_proportion=args.warmup_proportion,
    weight_decay=args.weight_decay,
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
    num_epoch=args.num_epoch,
    batch_size=args.batch_size,
    strategy=strategy)

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
#run_states = cls_task.finetune_and_eval()

#预测
(data,id) = GetFileRecord('./classify/classify_test.tsv')
result = cls_task.predict(data=data, return_result=True)
#存结果
submit = pd.read_csv('./classify/result.tsv',sep='\t',header=None)
for i in range(len(submit)) :
  submit.iloc[i,2] = result[i]
submit.to_csv('./classify/submit.tsv',sep='\t',header=None, index=False)
print(submit.head()) 
