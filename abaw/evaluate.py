import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast

def evaluate(config, model, eval_dataloader):
    with torch.no_grad():
        preds, labels = predict(config, model, eval_dataloader)
        r = torch.cov(torch.cat([preds, labels], dim=1)) / torch.std(preds) / torch.std(labels) # r = cov(x,y)/sd(x)/sd(y)
        r = r.mean()
    return r.cpu().numpy()


def predict(train_config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    preds = []
    labels = []
    with torch.no_grad():

        for wav2vec, vit, label in bar:

            with autocast():

                wav2vec = wav2vec.to(train_config.device)
                vit = vit.to(train_config.device)
                label = label.to(train_config.device)
                pred = model(wav2vec, vit)

            # save features in fp32 for sim calculation
            labels.append(label)
            preds.append(pred.to(torch.float32))

        # keep Features on GPU
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

    if train_config.verbose:
        bar.close()

    return preds, labels
