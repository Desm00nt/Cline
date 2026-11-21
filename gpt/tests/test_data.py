import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.data import PackedDataset, SFTDataset, sft_collate, write_bin
from minigpt.tokenizer import ASSISTANT_ID, END_ID, EOS_ID, USER_ID, BPETokenizer


def test_packed_dataset_windows(tmp_path):
    ids = list(range(100))
    p = str(tmp_path / "train.bin")
    write_bin(ids, p, vocab_size=1000)
    ds = PackedDataset(p, block_size=10)
    assert len(ds) == 9                       # (100 - 1 - 10) // 10 + 1
    x, y = ds[0]
    assert x.shape == (10,) and y.shape == (10,)
    assert x.tolist() == list(range(10))
    assert y.tolist() == list(range(1, 11))   # сдвиг на один токен
    x_last, _ = ds[len(ds) - 1]
    assert x_last.tolist() == list(range(80, 90))   # последнее полное окно


def test_packed_dataset_stride(tmp_path):
    ids = list(range(50))
    p = str(tmp_path / "train.bin")
    write_bin(ids, p, vocab_size=1000)
    ds = PackedDataset(p, block_size=10, stride=5)
    assert len(ds) == (50 - 1 - 10) // 5 + 1
    x, _ = ds[1]
    assert x.tolist() == list(range(5, 15))


def test_packed_dataset_rejects_short_corpus(tmp_path):
    p = str(tmp_path / "tiny.bin")
    write_bin([1, 2, 3], p, vocab_size=1000)
    with pytest.raises(ValueError):
        PackedDataset(p, block_size=16)


def test_write_bin_rejects_oversized_ids(tmp_path):
    # 65536 -> uint16, токен 70000 в него не влезает
    with pytest.raises(ValueError):
        write_bin([70000], str(tmp_path / "x.bin"), vocab_size=65536)
    # и токен вне словаря тоже должен быть отвергнут
    with pytest.raises(ValueError):
        write_bin([500], str(tmp_path / "y.bin"), vocab_size=400)


@pytest.fixture()
def tok():
    return BPETokenizer.train(["привет как дела", "спасибо всё хорошо",
                               "расскажи про модель", "модель учится на данных"],
                              num_merges=80, verbose=False)


def make_jsonl(path, dialogues):
    with open(path, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps({"messages": d}, ensure_ascii=False) + "\n")


def test_sft_masks_prompt(tmp_path, tok):
    path = str(tmp_path / "sft.jsonl")
    make_jsonl(path, [
        [{"role": "user", "content": "привет"},
         {"role": "assistant", "content": "привет как дела"}],
    ])
    ds = SFTDataset(path, tok, block_size=64)
    item = ds[0]
    assert item["x"].shape == item["y"].shape == item["mask"].shape
    # маска = 1 ровно на токенах ответа ассистента + завершающий <|end|>
    assert item["mask"].sum() > 0
    assert float(item["mask"][0]) == 0.0
    n_assistant = item["mask"].sum().item()
    body = tok.encode("привет как дела")
    assert n_assistant == len(body) + 1        # контент + <|end|>


def test_sft_targets_are_shifted(tmp_path, tok):
    path = str(tmp_path / "sft.jsonl")
    make_jsonl(path, [[{"role": "user", "content": "а"},
                       {"role": "assistant", "content": "б"}]])
    ds = SFTDataset(path, tok, block_size=64)
    item = ds[0]
    # y — это x со сдвигом на один токен: y[i] = x[i+1]
    assert torch.equal(item["x"][1:], item["y"][:-1])
    ids, mask = SFTDataset._render([{"role": "user", "content": "а"},
                                    {"role": "assistant", "content": "б"}], tok)
    assert ids[0] == EOS_ID
    assert ids[1] == USER_ID
    assert ASSISTANT_ID in ids and ids[-1] == END_ID
    assert mask[0] == 0 and mask[-1] == 1


def test_sft_truncates_to_block(tmp_path, tok):
    path = str(tmp_path / "sft.jsonl")
    make_jsonl(path, [[{"role": "user", "content": "слово " * 50},
                       {"role": "assistant", "content": "ответ"}]])
    ds = SFTDataset(path, tok, block_size=16)
    item = ds[0]
    assert item["x"].shape[0] == 16


def test_sft_collate_pads():
    batch = [
        {"x": torch.tensor([1, 2, 3]), "y": torch.tensor([2, 3, 4]), "mask": torch.tensor([0., 1., 1.])},
        {"x": torch.tensor([5, 6]), "y": torch.tensor([6, 7]), "mask": torch.tensor([1., 1.])},
    ]
    out = sft_collate(batch)
    assert out["x"].shape == (2, 3)
    assert out["x"][1, 2].item() == 0          # PAD_ID
    assert out["y"][1, 2].item() == -100       # игнорируется в cross_entropy
    assert out["mask"][1, 2].item() == 0.0


def test_system_prompt_injected(tmp_path, tok):
    path = str(tmp_path / "sft.jsonl")
    make_jsonl(path, [[{"role": "user", "content": "привет"},
                       {"role": "assistant", "content": "здравствуй"}]])
    ds = SFTDataset(path, tok, block_size=64, system="ты полезный ассистент")
    ids, _ = SFTDataset._render([{"role": "system", "content": "ты полезный ассистент"},
                                 {"role": "user", "content": "привет"},
                                 {"role": "assistant", "content": "здравствуй"}], tok)
    assert 5 in ids  # SYSTEM_ID
