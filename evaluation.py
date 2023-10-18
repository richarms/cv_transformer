import pickle
import os
import torch


def save_model(train_dic, results_dic, PATH):
    """
    Method to save model and results in PATH
    Input: train_dic: dictionary with model, optimizer and all trainingparameter
           results_dic: dictionary with results
           PATH: Path where to save the results
    Saves 4 files in folder date_time in the given PATH:
    model.p : contains the state dict of the model
    optimizer.p : contains the state dict of the optimizer
    results_summary.txt : contains import results and parameters in a textfile
    summary_results.p : contains all results and all parameters for the model as a dictionary
    """
    # save state dicts of model and optimizer
    torch.save(train_dic["model"].state_dict(), os.path.join(PATH, 'model.pt'))
    torch.save(train_dic["optimizer"].state_dict(), os.path.join(PATH, 'optimizer.pt'))

    # save important results in textfile
    with open(os.path.join(PATH, 'results_summary.txt'), 'w') as results:
        results.write('Results of Training for Sirt:')
        results.write('\n')
        results.write('Epochs: {}, Batch_size: {}, Optimizer: {}, Trained on {}'.format(train_dic["epochs"], train_dic["batch_size"], type(train_dic["optimizer"]).__name__, train_dic["device"]))
        results.write('\n')
        results.write('Training complete in {:.0f}m {:.0f}s'.format(results_dic["epoch_time"] // 60, results_dic["epoch_time"] % 60))
        results.write('\n')
        results.write('Training loss after Training: {} ({})'.format(results_dic["train_loss"][-1], type(train_dic["loss"]).__name__))
        # results.write('\n')
        # results.write('Training accuraty after Training: {}'.format(results_dic["train_acc"][-1]))
        results.write('\n')
        results.write('Validation loss after Training: {} ({})'.format(results_dic["val_loss"][-1], type(train_dic["loss"]).__name__))
        results.write('\n')
        results.write('Validation accuracy after Training: {}'.format(results_dic["val_acc"][-1]))

    # add modelparameter and results to summary_dic
    summary_dic = results_dic
    summary_dic["batch_size"] = train_dic["batch_size"]
    summary_dic["epochs"] = train_dic["epochs"]
    summary_dic["device"] = train_dic["device"]
    summary_dic["loss"] = train_dic["loss"]

    # save summary_dic as dictionary
    with open(os.path.join(PATH, 'summary_results.p'), 'wb') as summary_results:
        pickle.dump(summary_dic, summary_results, protocol=pickle.HIGHEST_PROTOCOL)

def save_model_test(results_dic, PATH):
    """
    Method to save testresults in PATH
    Input: results_dic: dictionary with results
           PATH: Path where to save the results
    results_summary.txt : contains import results and parameters in a textfile
    """

    # save important results in textfile
    with open(os.path.join(PATH, 'results_summary.txt'), 'w') as results:
        results.write('Results of Testing:')
        results.write('\n')
        results.write('Test loss after Training: {}'.format(results_dic["test_loss"][-1]))
        results.write('\n')
        results.write('Test accuracy after Training: {}'.format(results_dic["test_acc"][-1]))

def load_model(model, opt, model_path):
    # inplace operation for model and opt
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
    # TODO: some problem loading the optimitzer, gets loaded to cpu even with map_location
    opt.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
    # return model, opt
