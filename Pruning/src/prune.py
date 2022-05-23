# -*- coding: utf-8 -*-
"""
Created on Mon May 23 00:12:19 2022

@author: harsh
"""

from torch.nn.utils import prune
import seaborn as sns
class prune_model():
  def __init__(self, loader, model_path, prune_type='l1_unstructured'):
    self.loader = loader
    self.prune_type = prune_type
    

  def get_model(self, model_path = "/content/drive/MyDrive/MaskDetectionDataset/models/cnn_scratch.pt"):
    """
    loads the saved model 
    """
    #model_path = "/content/drive/MyDrive/MaskDetectionDataset/models/cnn_scratch.pt"
    model1 = FaceMaskClassifier2()
    model1.load_state_dict(torch.load(model_path))
    self.model = model1
    return model1

  def get_model_modules(self, mdl):
    layers = []

    def unwrap_recur(modules):
      for md in modules.children():
        if isinstance(md, nn.Sequential):
          unwrap_recur(md)
        elif isinstance(md, nn.ModuleList):
          for m in md:
            unwrap_recur(m)
        else:
          layers.append(md)

    unwrap_recur(mdl)

    return nn.ModuleList(layers)

  def prune_model_l1_unstructured(self, model, ly_type, ratio):
    """
    Model pruning
    """
    modules = get_model_modules(model)
    for m in modules.children():
      if isinstance(m, ly_type):
        prune.l1_unstructured(m, 'weight', ratio)
        prune.remove(m, 'weight')
    return model

  def prune_model_l1_structured(self, model, ly_type, ratio):
    """
    Model pruning
    """
    modules = get_model_modules(model)
    for m in modules.children():
      if isinstance(m, ly_type):
        prune.ln_structured(m, 'weight', ratio, n=1, dim=1)
        prune.remove(m, 'weight')
    return model

  def get_eval(self, model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval().to(device)
    val_loss = []
    val_epoch_loss = 0.0
    val_epoch_acc = 0.0
    with torch.no_grad():
      for X_val_batch, y_val_batch in loader:
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
        #y_val_pred = model(X_val_batch).squeeze()
        y_val_pred = model(X_val_batch).squeeze()
        val_loss = criterion(y_val_pred, y_val_batch)
        val_acc = binary_acc(y_val_pred, y_val_batch)
        val_epoch_loss += val_loss.item()
        val_epoch_acc += val_acc.item()

    print("Accuracy: "+str(val_epoch_acc / len(loader))+" Loss : " + str(val_epoch_loss / len(loader)))
    return val_epoch_acc / len(loader), val_epoch_loss / len(loader)

  def prune(self, model_, prune_fn, eval_fn):
    losses = []
    accuracy = []

    print("Evaluating the model with 0 pruning")
    model = model_()
    acc, loss = eval_fn(model, self.loader)
    losses.append((0, loss))
    accuracy.append((0, acc))

    # Pruning 
    for i in range(1, 18):
      ratio = i*0.05
      print(f"Evaluating model with {ratio} pruning")
      model = model_()
      pruned_model = prune_fn(model, nn.Conv2d, ratio)
      acc, loss = eval_fn(pruned_model, self.loader)
      losses.append((ratio, loss))
      accuracy.append((ratio, acc))
    
    return accuracy, losses
  
  def plot(self, acc, losses, title):
    (pd.DataFrame(losses, columns=['sparsity', 'loss'])
    .pipe(lambda df: df.assign(
        perf=(df.loss - pd.Series([losses[0][1]] * len(df))) / losses[0][1] + 1
    ))
    .head(10)
    .plot.line(x='sparsity', y='perf', figsize=(12, 8), title=title)
    )
    sns.despine()

  def fit(self):
    if self.prune_type == "l1_unstructured":
      accs, losses = self.prune(self.get_model, self.prune_model_l1_unstructured, self.get_eval)
      #self.plot(accs, loss, "L1_Ustructured")
    else:
      accs, losses = self.prune(self.get_model, self.prune_model_l1_structured, self.get_eval)
      #self.plot(accs, loss, "L1_structured")
    return accs, losses
