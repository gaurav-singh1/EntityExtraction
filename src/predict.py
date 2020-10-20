
import numpy as np
import torch 



import config
import dataset
import engine
from model import EntityModel

import joblib

if __name__ =="__main__":
    meta_data = joblib.load('meta.bin')
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]


    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    sentence = "momo is going to london"

    tokenized_sentence = config.TOKENIZER.encode(sentence)
    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)


    test_dataset = dataset.EntityDataset(
        texts = [sentence], 
        pos = [[0] * len(sentence)], 
        tags = [[0] * len(sentence)]
    )

    device = torch.device('cpu')
    model = EntityModel(num_pos=num_pos, num_tag = num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.todevice(v).unsqueeze(0)
        tag, pos, _ = model(**data)

        print(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence)]
        print(enc_pos.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence)]

