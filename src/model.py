import config
import torch
import torch.nn as nn
import transformers

def loss_fn(outputs, targets, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = outputs.view(-1, num_labels)
    active_labels = torch.where(active_loss
    , targets.view(-1), torch.tensor(lfn.ignore_index).type_as(targets))
    loss = lfn(active_logits, active_labels)

    return loss
 


class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)
        self.bert_drop1 = nn.Dropout(0.3)
        self.bert_drop2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_pos, target_tags):
        o1, _ = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        bo_tag =self.bert_drop1(o1)
        bo_pos =self.bert_drop2(o1)
        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos)

        loss_tag = loss_fn(tag, target_tags, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_pos + loss_tag) / 2
        
        return tag, pos, loss





