import os
# import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from sklearn.metrics import average_precision_score
from datetime import datetime

from torch.utils.data import DataLoader
from dl_signal.utils import SignalDataset_music
from attention.real_transformer import real_Transformer_Pred
from train_real_transformer import train, test_evaluate

def execution(load_path=None, train_model=True, best=False, current_date_time=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    torch.manual_seed(42)
    batch_size = 35

    root_dir = "/data/home/f_eile01/Projects/complex_nn/complex_transformer/dl_signal/music/musicnet/"

    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    train_ds = SignalDataset_music(os.path.join(root_dir, 'train'), time_step=64, train='train')
    valid_ds = SignalDataset_music(os.path.join(root_dir, 'valid'), time_step=64, train='valid')
    test_ds = SignalDataset_music(os.path.join(root_dir, 'test'), time_step=64, train='test')
    if current_date_time is None:
        current_date_time = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    num_train = len(train_ds)
    num_valid = len(valid_ds)
    num_test = len(test_ds)

    print(f'Training samples: {num_train}, valid samples: {num_valid}, test samples: {num_test}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    input_dim = train_ds[0][0].shape[1]
    output_dim = train_ds[0][1].shape[1]
    tokens = train_ds[0][0].shape[0]

    model = real_Transformer_Pred(input_dim=input_dim, output_dim=output_dim, embed_dim=320, hidden_dim=2048, num_heads=8, layers=6, tokens=tokens)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if load_path is not None:
        model = model.to(device)
        for x, y in train_dl:
            x = x.to(device)
            x = x.type(torch.complex64)
            model(x)
            break

        model.load_state_dict(torch.load(os.path.join(load_path, 'model.pt')))
        optimizer.load_state_dict(torch.load(os.path.join(load_path, 'optimizer.pt')))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    loss_fn = nn.BCEWithLogitsLoss()
    acc_fn = average_precision_score

    if train_model:
        trained_model = train(model, train_dl, valid_dl, valid_ds, loss_fn, acc_fn, optimizer, scheduler, current_date_time, config, epochs=100)

    test_evaluate(model, test_dl, test_ds, loss_fn, acc_fn, current_date_time, config, best=best)


if __name__ == "__main__":
    execution()
