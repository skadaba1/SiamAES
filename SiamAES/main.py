import os
import pickle
from torch.utils.data import DataLoader
from trainer_classes import Trainer, SiameseTrainer, TripletTrainer
from data import gen_data
from utils import load_and_filter, tokenize
from evals import run_summary
from model import BaselineModel, SiameseNetwork, SiameseNetworkLSTM
import torch
from utils import device, generate_figures, grad_explain


def __main__(num_classes, n_each, n_augment, dirpath, flag, model_flag, num_epochs, lr):
    prompt = '7'
    splits = [25, 50]
    
    # Generate data
    num_classes, counts, train_dataloader, eval_dataloader, test_dataloader, essays, batches_train, batches_test = gen_data(prompt, num_classes, n_each, n_augment, flag, splits)
    print("\nTarget number of classes =", num_classes)
    print(counts, "\n")

    # Define and load pretrained model
    if flag == 0:
        model = BaselineModel(num_classes)
    elif flag in [1, 2] and model_flag == 0:
        model = SiameseNetwork(num_classes)
    else:
        model = SiameseNetworkLSTM(num_classes)
    
    model = model.to(device)

    # Initialize the appropriate trainer class
    if flag == 0:
        trainer = Trainer(model, train_dataloader, eval_dataloader, device, lr=lr, num_epochs=num_epochs)
    elif flag == 1:
        trainer = SiameseTrainer(model, train_dataloader, eval_dataloader, device, lr=lr, num_epochs=num_epochs)
    elif flag == 2:
        trainer = TripletTrainer(model, train_dataloader, eval_dataloader, device, lr=lr, num_epochs=num_epochs)
    else:
        raise ValueError("Invalid flag value")

    # Train the model
    train_loss_history, val_loss_history = trainer.train()

    # Evaluate the model
    accuracy, f1, confusion, clusters, labels, predicted, grads = run_summary(model, test_dataloader, num_classes, flag, model_flag)
    print("Accuracy =", accuracy, "| f1 =", f1)

    # Save metadata
    meta = {
        'accuracy': accuracy,
        'f1': f1,
        'grads': grads,
        'essays': essays,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'num_classes': num_classes,
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'predictions': predicted,
        'labels': labels
    }
    fname = os.path.join(dirpath, "summary.pickle")
    with open(fname, 'wb') as handle:
        pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate and save figures
    train_fig, validation_fig, confusion_fig, clusters_fig = generate_figures(
        train_loss_history, val_loss_history, confusion, clusters, labels, dirpath, accuracy, f1
    )

    # Grad explanation if applicable
    if flag and not model_flag:
        k, n = 9, 1
        grad_explain(grads, essays, dirpath, k, n)

    return 0

if __name__ == "__main__":
    __main__(num_classes=4, n_each=5, n_augment=5, dirpath='.', flag=0, model_flag=0, num_epochs=5, lr=0.001)
