import torch
from tqdm import tqdm

def train_fn(data_loader, model, device, optimizer, scheduler):
    model.train()
    final_loss = 0
    for bi, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        
        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        final_loss+=loss.item()
    
    return final_loss / len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, len = len(data_loader)):
        for k, v in data.items():
            data[k] = v.todevice(v)
        
        _, _, loss = model(**data)
        final_loss+=loss.item()
    
    return final_loss / len(data_loader)

