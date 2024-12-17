## Evaluation ##
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn import decomposition
from sklearn.cluster import OPTICS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from matplotlib.figure import Figure
from io import BytesIO
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_eval(model, test_dataloader, flag, model_flag):
    # on test set ##
    model.eval()
    predictions_test = []
    labels_test = []
    E = []
    softmax = nn.Softmax(dim=1)
    grads = []
    for batch_i, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
    
        if(flag):
          embed_x1, update_x1= model(batch, None, None)
          logits = update_x1[0]; target = update_x1[1]
          if(flag and not model_flag):
            grad = update_x1[3][0].detach().cpu().numpy()
          else:
            grad = []
          grads.append(grad)
        else:
          outputs = model(**batch); 
          logits = outputs[0]['logits']; embed_x1 = outputs[1]; target = outputs[2]
        
        pred = np.argmax(softmax(logits.detach().cpu()))
        truth = np.argmax(target.detach().cpu())
        predictions_test.append(pred)
        labels_test.append(truth)
        E.append(embed_x1)

    E = torch.stack(E, dim=0)
    E = E.view(len(test_dataloader), -1).detach().cpu()

    pca = decomposition.PCA(n_components=2)
    x = torch.Tensor(pca.fit_transform(E))
    clusters=x
    return labels_test, predictions_test, clusters, grads

def get_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
      print(predicted[i], actual[i])
      if(predicted[i] == actual[i]):
        correct += 1
    return correct/len(actual)

def get_f1(actual, predicted, num_classes):
  f1s = []
  for i in range(num_classes):
    tp = 0
    fp = 0
    fn = 0
    for j in range(len(predicted)):
      if(predicted[j] == actual[j] and actual[j] == i):
        tp += 1
      if(predicted[j] == i and actual[j] != i):
        fp += 1
      if(predicted[j] != i and actual[j] == i):
        fn += 1
    precision = tp / (tp + fp + 1)
    recall = tp / (tp + fn + 1)
    f1 = 2*precision*recall / (precision+recall+1)
    f1s.append(f1)
  return f1s

def get_confusion(actual, predicted):
    return metrics.confusion_matrix(actual, predicted)

def text_to_rgba(s, *, dpi, **kwargs):
    # To convert a text string to an image, we can:
    # - draw it on an empty and transparent figure;
    # - save the figure to a temporary buffer using ``bbox_inches="tight",
    #   pad_inches=0`` which will pick the correct area to save;
    # - load the buffer using ``plt.imread``.
    #
    # (If desired, one can also directly save the image to the filesystem.)
    fig = Figure(facecolor="none")
    fig.text(0, 0, s, **kwargs)
    with BytesIO() as buf:
        fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)
        rgba = plt.imread(buf)
    return rgba


def run_summary(model, test_dataloader, num_classes, flag, model_flag):

   actual, predicted, clusters, grads = run_eval(model, test_dataloader, flag, model_flag)
   accuracy = get_accuracy(actual, predicted)
   f1 = get_f1(actual, predicted, num_classes)
   confusion = get_confusion(actual, predicted)

   return accuracy, f1, confusion, clusters, actual, predicted, grads

def generate_figures(training_history, validation_history, confusion_matrix, clusters, labels, dirpath, accuracy, f1):
    x = range(len(training_history))
    fig1 = plt.figure(1)
    plt.plot(x, training_history)
    plt.xlabel("Epoch #")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training History")
    plt.savefig(dirpath+'/train_history.png')
    
    x = range(len(validation_history))
    fig2 = plt.figure(2)
    plt.plot(x, validation_history)
    plt.xlabel("Epoch #")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Validation History")
    plt.savefig(dirpath+'/validation_history.png')

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    cax = ax.matshow(confusion_matrix, interpolation='nearest')
    fig3.colorbar(cax)
    plt.savefig(dirpath+'/confusion_matrix.png')

    fig4 = plt.figure(4)
    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(clusters[:, 0], clusters[:, 1], c=labels, cmap='cool')
    plt.tight_layout()
    plt.savefig(dirpath+'/clusters.png')
    plt.show()

    fig5 = plt.figure(5)
    rgba1 = text_to_rgba("Accuracy = " + str(accuracy), color="blue", fontsize=5, dpi=200)
    rgba2 = text_to_rgba("F1 = " + str(f1), color="red", fontsize=5, dpi=200)
    # One can then draw such text images to a Figure using `.Figure.figimage`.
    fig5.figimage(rgba1, 25, 350)
    fig5.figimage(rgba2, 25, 250)
    plt.savefig(dirpath+'/stats.png')
    
    return fig1, fig2, fig3, fig4

def grad_explain(grads, essays, dirpath, k, n=1):
  for i in range(n):
    pos_weight = np.sum(grads[i], axis=2)[0][0:k]
    pos_weight = (pos_weight - np.min(pos_weight)) * (1/(np.max(pos_weight) - np.min(pos_weight)))
    essay = essays[i].split()
    labels = (np.asarray(["{}".format(value)
                        for value in essay[0:k]])
          ).T
    labels = np.expand_dims(labels,1).T

    fig, ax = plt.subplots(figsize=(15, 2))
    ax0 = plt.axes()
    ax1 = sns.heatmap(np.expand_dims(pos_weight,1).T, annot=labels, cbar=0, cmap="YlGnBu",linewidths=5, ax=ax0, fmt="")
    plt.savefig(dirpath+'/gradient_weights.png')
    plt.show()
