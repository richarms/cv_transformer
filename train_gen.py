import os
# import matplotlib.pyplot as plt
import torch
import time
from evaluation import save_model, save_model_test

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device.".format(device))
torch.manual_seed(42)


def train(model, train_dl, valid_dl, valid_ds, loss_fn, acc_fn, optimizer, scheduler, current_date_time, config, sm_variante='realip', device="cuda", epochs=1):
    start = time.time()
    model.train()
    model = model.to(device)
    train_loss = []
    valid_loss = []
    valid_acc = []
    escape = False
    set_y = False

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                dataloader = train_dl

            else:
                model.train(False)
                dataloader = valid_dl
            running_loss = 0.0
            all_batches = 0

            step = 0
            shape = [len(valid_ds), 21, 128]  # shape of predictions/trues
            preds = torch.zeros(shape)
            trues = torch.zeros(shape)

            for j, (data, label) in enumerate(dataloader):

                num_input = 43
                num_gen = 64 - num_input

                x_data = data[:, :num_input].to(device)
                x_data = x_data.type(torch.complex64)
                y_data = data[:, num_input:].to(device)
                y_data = y_data.type(torch.complex64)
                y_label = label[:, num_input:].to(device).type(torch.complex64)  # save as complex tensor, to be processed, imag is still 0

                step += 1
                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(x_data, tgt=y_label)
                    loss = loss_fn(outputs, y_label.real)  # imag of y_label is 0, real does not change anything, just the processing
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.35)
                    # for param in model.parameters():
                    #     print(torch.norm(param))
                    loss.backward()

                    optimizer.step()

                else:
                    with torch.no_grad():
                        model.eval()
                        outputs = torch.zeros(y_label.shape, device=device)
                        for i in range(num_gen):
                            outputs[:, i] = model(x_data, tgt=outputs)[:, i]
                        loss = loss_fn(outputs, y_label.real)

                        trues[j * dataloader.batch_size: (j + 1) * dataloader.batch_size, :, :] = y_label.real.detach()
                        preds[j * dataloader.batch_size: (j + 1) * dataloader.batch_size, :, :] = outputs.detach()

                running_loss += loss.detach() * dataloader.batch_size
                all_batches += dataloader.batch_size

                if step % 1 == 0:
                    print('Current step: {} Loss {} AllocMem(Mb): {}'.format(step, loss, torch.cuda.memory_allocated() / 1024 / 1024))
            epoch_loss = running_loss / all_batches

            print('{} Loss {:.4f}'.format(phase, epoch_loss))
            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

            if phase == 'valid':
                epoch_acc = acc_fn(trues.flatten().cpu(), torch.sigmoid(preds.flatten().cpu()))
                scheduler.step(epoch_loss)
                print('{} accuracy {:.4f}'.format(phase, epoch_acc))
                valid_acc.append(epoch_acc)

                time_elapsed = time.time() - start

                results_dict = {"epoch_time": time_elapsed,
                                "train_loss": train_loss,
                                "val_loss": valid_loss,
                                "val_acc": valid_acc}
                train_dict = {"model": model,
                              "optimizer": optimizer,
                              "loss": loss_fn,
                              "device": device,
                              "batch_size": dataloader.batch_size,
                              "epochs": epochs}

                full_save_path = os.path.join(config["model_path"], train_dict["model"].name, sm_variante, current_date_time, str(epoch))
                os.makedirs(full_save_path, exist_ok=True)
                save_model(train_dict, results_dict, full_save_path)
    time_elapsed = time.time() - start

    print('Training complete in {:.0f}m, {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


def test_evaluate(model, test_dl, test_ds, loss_fn, acc_fn, current_date_time, config, sm_variante='realip', device="cuda", best=False):
    model = model.to(device)
    test_loss = []
    test_acc = []

    model.train(False)
    dataloader = test_dl
    running_loss = 0.0
    all_batches = 0

    step = 0
    shape = [len(test_ds), 21, 128]
    preds = torch.zeros(shape)
    trues = torch.zeros(shape)

    for j, (data, label) in enumerate(dataloader):

        num_input = 43
        num_gen = 64 - num_input

        x_data = data[:, :num_input].to(device)
        x_data = x_data.type(torch.complex64)
        y_data = data[:, num_input:].to(device)
        y_data = y_data.type(torch.complex64)
        y_label = label[:, num_input:].to(device).type(torch.complex64)  # save as complex tensor, to be processed, imag is still 0

        step += 1
        with torch.no_grad():
            model.eval()
            outputs = torch.zeros(y_label.shape, device=device)
            for i in range(num_gen):
                outputs[:, i] = model(x_data, tgt=outputs)[:, i]
            loss = loss_fn(outputs, y_label.real)

            trues[j * dataloader.batch_size: (j + 1) * dataloader.batch_size, :, :] = y_label.real.detach()
            preds[j * dataloader.batch_size: (j + 1) * dataloader.batch_size, :, :] = outputs.detach()

        running_loss += loss.detach() * dataloader.batch_size
        all_batches += dataloader.batch_size

        if step % 1 == 0:
            print('Current step: {} Loss {} AllocMem(Mb): {}'.format(step, loss, torch.cuda.memory_allocated() / 1024 / 1024))

    epoch_loss = running_loss / all_batches
    epoch_acc = acc_fn(trues.flatten().cpu(), torch.sigmoid(preds.flatten().cpu()))

    print('Test Loss {:.4f}'.format(epoch_loss))
    test_loss.append(epoch_loss)

    print('Test accuracy {:.4f}'.format(epoch_acc))
    test_acc.append(epoch_acc)

    results_dict = {"test_loss": test_loss,
                    "test_acc": test_acc}

    if best is True:
        full_save_path = os.path.join(config["model_path"], model.name, sm_variante, current_date_time, 'test_results_best')
    else:
        full_save_path = os.path.join(config["model_path"], model.name, sm_variante, current_date_time, 'test_results')
    os.makedirs(full_save_path, exist_ok=True)
    save_model_test(results_dict, full_save_path)
