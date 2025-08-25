"""Train a multi-label text classifier on transcript + OCR text.

This script expects a JSONL dataset where each row has `text` fields and a
`flags` mapping of label -> bool. See `scripts/label_dataset.py` for
producing labeled examples.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

LABELS = [
    "missing_disclaimer",
    "offshore_reference",
    "risk_free",
    "chasing_losses",
    "solve_financial_problems",
    "vpn_proxy",
]


def load_dataset(path: Path) -> Dataset:
    records: List[Dict[str, object]] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            text = "{}\n{}".format(row.get("transcript", ""), row.get("ocr_text", ""))
            flags = row.get("flags", {})
            record = {"text": text}
            for lbl in LABELS:
                record[lbl] = int(bool(flags.get(lbl)))
            records.append(record)
    return Dataset.from_list(records)


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a multi-label classifier")
    p.add_argument("dataset", type=Path, help="Path to labeled JSONL dataset")
    p.add_argument("output", type=Path, help="Directory to store the trained model")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    ds = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess(batch: Dict[str, str]) -> Dict[str, object]:
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
        tokens["labels"] = [batch[lbl] for lbl in LABELS]
        return tokens

    tokenized = ds.map(preprocess)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(LABELS), problem_type="multi_label_classification"
    )

    args_train = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(model=model, args=args_train, train_dataset=tokenized, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(str(args.output))


if __name__ == "__main__":
    main()
