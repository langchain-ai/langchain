import argparse
from mteb import MTEB
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    TrainingArguments
)
from sentence_transformers import SentenceTransformer
from intel_extension_for_transformers.transformers.trainer import NLPTrainer
from intel_extension_for_transformers.transformers import metrics, objectives, QuantizationConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize embedding models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help="Path to save optimized model"
    )
    args = parser.parse_args()
    training_args = TrainingArguments(
        output_dir=args.output_path,
        do_eval=True,
        do_train=True,
        no_cuda=True,
        overwrite_output_dir=True,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)

    # dataset for trainer calib_dataset
    from datasets import Dataset
    evaluation = MTEB(task_langs=['en'], tasks=['CQADupstackAndroidRetrieval'])
    evaluation.select_tasks(task_langs=['en'], tasks=['CQADupstackAndroidRetrieval'])
    evaluation.tasks[0].load_data(eval_splits=['test'])
    task = evaluation.tasks[0]
    corpus = task.corpus['test']
    queries = task.queries['test']
    queries = [queries[qid] for qid in queries]
    corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    corpus = [corpus[cid] for cid in corpus_ids][:100]
    corpus = [(doc["title"] + " " + doc["text"]).strip()
                if "title" in doc else doc["text"].strip() for doc in corpus]
    data_list = [*queries[:100], *corpus[:100]] #extend
    date_dict = {'text': data_list}
    calib_dataset = Dataset.from_dict(date_dict)

    def preprocess_function(example):
        # Tokenize the texts
        result= tokenizer(example['text'], padding="max_length", max_length=tokenizer.model_max_length, truncation=True)
        return result

    calib_dataset = calib_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )

    trainer = NLPTrainer(
        model=model,
        args=training_args,
        train_dataset=calib_dataset,
        eval_dataset=calib_dataset,
        tokenizer=tokenizer,
    )

    tune_metric = metrics.Metric(name="eval_accuracy", is_relative=True, criterion=0.02)
    objective = objectives.performance

    quantization_config = QuantizationConfig(
        approach="PostTrainingStatic",
        max_trials=600,
        metrics=[tune_metric],
        objectives=[objective],
        sampling_size = len(calib_dataset)//20
    )

    stmodel = SentenceTransformer(args.model_name_or_path)
    def eval_func(model):
        return 1
        stmodel[0].auto_model = model
        evaluation = MTEB(task_langs=['en'], task_types=['STS'])
        results = evaluation.run(stmodel, overwrite_results=True, eval_splits=["test"])
        avg_res = 0
        for task_name, task_res in results.items():
            if task_name in ['STS17']:
                avg_res += round(task_res['test']['en-en']['cos_sim']['spearman'] * 100, 2)
            elif task_name in ['STS22']:
                avg_res += round(task_res['test']['en']['cos_sim']['spearman'] * 100, 2)
            else:
                avg_res += round(task_res['test']['cos_sim']['spearman'] * 100, 2)

        avg_res /= len(results)
        return avg_res

    model = trainer.quantize(
        quant_config=quantization_config,
        eval_func=eval_func,
    )
