from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from inspect import signature

from transformers import (
    DataCollatorForTokenClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

from .layout_transformers import (
    BODY_PREFIX,
    DEFAULT_MODEL_ID,
    HEADER_PREFIX,
    OTHER_LABEL,
    _load_layoutlmv3,
)

log = logging.getLogger(__name__)


def _read_jsonl(paths: Sequence[str]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path_str in paths:
        path = Path(path_str).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el dataset: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def _collect_label_set(records: Iterable[Dict[str, object]]) -> List[str]:
    label_set = set()
    for rec in records:
        for lbl in rec.get("labels", []):
            label_set.add(lbl)
    label_set.add(OTHER_LABEL)
    # Garantizar orden consistente: header cols, body cols, other
    def _key(label: str) -> Tuple[int, str]:
        if label.startswith(HEADER_PREFIX):
            return (0, label)
        if label.startswith(BODY_PREFIX):
            return (1, label)
        if label == OTHER_LABEL:
            return (2, label)
        return (3, label)

    return sorted(label_set, key=_key)


@dataclass
class LayoutLMSample:
    image_path: str
    words: List[str]
    boxes: List[List[int]]
    labels: List[str]


class LayoutLMDataset(Dataset):
    def __init__(
        self,
        samples: List[LayoutLMSample],
        processor: LayoutLMv3Processor,
        label2id: Dict[str, int],
        max_seq_length: int = 512,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.label2id = label2id
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        labels = [self.label2id.get(lbl, self.label2id[OTHER_LABEL]) for lbl in sample.labels]
        encoding = self.processor(
            image,
            sample.words,
            boxes=sample.boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        # processor devuelve tensores con dimensión batch=1
        return {key: tensor.squeeze(0) for key, tensor in encoding.items()}


def _prepare_samples(records: List[Dict[str, object]]) -> List[LayoutLMSample]:
    samples: List[LayoutLMSample] = []
    for rec in records:
        samples.append(
            LayoutLMSample(
                image_path=str(rec["image_path"]),
                words=list(rec["words"]),
                boxes=[list(box) for box in rec["bboxes"]],
                labels=list(rec["labels"]),
            )
        )
    return samples


def _split_train_eval(
    records: List[Dict[str, object]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not records:
        return [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    eval_size = max(1, int(len(records) * eval_ratio)) if eval_ratio > 0 else 0
    eval_indices = set(indices[:eval_size])
    train_records = [records[i] for i in range(len(records)) if i not in eval_indices]
    eval_records = [records[i] for i in range(len(records)) if i in eval_indices]
    return train_records, eval_records


def _build_datasets(
    *,
    train_records: List[Dict[str, object]],
    eval_records: Optional[List[Dict[str, object]]],
    processor: LayoutLMv3Processor,
    label2id: Dict[str, int],
    max_seq_length: int,
) -> Tuple[LayoutLMDataset, Optional[LayoutLMDataset]]:
    train_ds = LayoutLMDataset(
        _prepare_samples(train_records),
        processor=processor,
        label2id=label2id,
        max_seq_length=max_seq_length,
    )
    eval_ds = None
    if eval_records:
        eval_ds = LayoutLMDataset(
            _prepare_samples(eval_records),
            processor=processor,
            label2id=label2id,
            max_seq_length=max_seq_length,
        )
    return train_ds, eval_ds


def _compute_metrics_builder(id2label: Dict[int, str]):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        total = 0
        correct = 0
        for pred_row, label_row in zip(predictions, labels):
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                total += 1
                if pred_id == label_id:
                    correct += 1
        accuracy = correct / total if total else 0.0
        return {"token_accuracy": accuracy}

    return _compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning de LayoutLMv3-base para tablas financieras.")
    parser.add_argument("--train-jsonl", nargs="+", required=True, help="Archivos JSONL de entrenamiento.")
    parser.add_argument("--eval-jsonl", nargs="+", help="Archivos JSONL de evaluación. Si no se especifican, se toma un split del train.")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Proporción del train que se usará como validación si no hay eval explícito.")
    parser.add_argument("--output-dir", required=True, help="Directorio donde se guardará el checkpoint fine-tuneado.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Checkpoint base de LayoutLMv3.")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", help="Ruta a checkpoint para reanudar entrenamiento.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Entrenar en mixed precision (requiere GPU).")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers para DataLoader.")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--metrics-json", help="Ruta opcional para guardar el historial de métricas (JSON).")
    parser.add_argument("--metrics-csv", help="Ruta opcional para guardar el historial de métricas (CSV).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.loglevel, format="%(asctime)s - %(levelname)s - %(message)s")

    train_records = _read_jsonl(args.train_jsonl)
    if args.eval_jsonl:
        eval_records = _read_jsonl(args.eval_jsonl)
    else:
        train_records, eval_records = _split_train_eval(train_records, args.eval_ratio, args.seed)

    label_list = _collect_label_set(train_records + (eval_records or []))
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    log.info("Etiquetas: %s", label_list)

    processor, _ = _load_layoutlmv3(args.model_id)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_id,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds, eval_ds = _build_datasets(
        train_records=train_records,
        eval_records=eval_records,
        processor=processor,
        label2id=label2id,
        max_seq_length=args.max_seq_length,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=processor.tokenizer,
        pad_to_multiple_of=8,
    )

    ta_params = set(signature(TrainingArguments.__init__).parameters.keys())
    def _set(name: str, value) -> None:
        if name in ta_params:
            training_kwargs[name] = value

    training_kwargs = dict(output_dir=args.output_dir)
    _set("num_train_epochs", args.num_epochs)
    _set("per_device_train_batch_size", args.batch_size)
    _set("per_device_eval_batch_size", args.batch_size)
    _set("learning_rate", args.learning_rate)
    _set("weight_decay", args.weight_decay)
    _set("warmup_ratio", args.warmup_ratio)
    _set("logging_steps", args.logging_steps)
    _set("save_steps", args.save_steps)
    _set("save_total_limit", 2)
    _set("seed", args.seed)
    _set("gradient_accumulation_steps", args.gradient_accumulation_steps)
    _set("dataloader_num_workers", args.num_workers)
    _set("fp16", args.fp16)
    if eval_ds:
        strategy_value = "steps"
    else:
        strategy_value = "no"

    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = strategy_value
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = strategy_value
    else:
        # Older versions rely on evaluate_during_training flag.
        if eval_ds and "evaluate_during_training" in ta_params:
            training_kwargs["evaluate_during_training"] = True

    if bool(eval_ds) and "load_best_model_at_end" in ta_params:
        training_kwargs["load_best_model_at_end"] = True

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=_compute_metrics_builder(id2label),
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log.info("Entrenamiento finalizado. Checkpoint guardado en %s", args.output_dir)

    history = trainer.state.log_history
    if args.metrics_json:
        path = Path(args.metrics_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        log.info("Historial de métricas guardado en JSON: %s", path)

    if args.metrics_csv:
        import csv

        path_csv = Path(args.metrics_csv).expanduser()
        path_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for record in history if isinstance(record, dict) for k in record.keys()})
        with path_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for record in history:
                if isinstance(record, dict):
                    writer.writerow({k: record.get(k) for k in keys})
        log.info("Historial de métricas guardado en CSV: %s", path_csv)


if __name__ == "__main__":
    main()
