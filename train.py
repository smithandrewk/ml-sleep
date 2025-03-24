# import torch
# import os
# from torch.utils.data import TensorDataset,ConcatDataset,DataLoader
# import matplotlib.pyplot as plt
# from torch.nn.functional import relu
# from sklearn.preprocessing import RobustScaler
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from ekyn import SleepStageClassifier

from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

import torch
from torch import nn

from lib.utils import get_dataloader
from lib.utils import SleepStageClassifier
from lib.utils import ekyn_ids,snezana_mice_ids,courtney_ids
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
from lib.utils import calculate_f1,plot_training_progress
batch_size = 1024
ekyn_ids = ekyn_ids[:8]
snezana_mice_ids = snezana_mice_ids[:8]
courtney_ids = courtney_ids[:8]
print(ekyn_ids,snezana_mice_ids,courtney_ids)

train_ekyn_ids,test_ekyn_ids = ekyn_ids[:-len(ekyn_ids)//4],ekyn_ids[-len(ekyn_ids)//4:]
print(len(train_ekyn_ids),len(test_ekyn_ids),train_ekyn_ids,test_ekyn_ids)
train_snezana_mice_ids,test_snezana_mice_ids = snezana_mice_ids[:-len(snezana_mice_ids)//4],snezana_mice_ids[-len(snezana_mice_ids)//4:]
print(len(train_snezana_mice_ids),len(test_snezana_mice_ids),train_snezana_mice_ids,test_snezana_mice_ids)
train_courtney_ids,test_courtney_ids = courtney_ids[:-len(courtney_ids)//4],courtney_ids[-len(courtney_ids)//4:]
print(len(train_courtney_ids),len(test_courtney_ids),train_courtney_ids,test_courtney_ids)

trainloader = get_dataloader(train_ekyn_ids,train_snezana_mice_ids,courtney_ids=train_courtney_ids,batch_size=batch_size,shuffle=True)
testloader = get_dataloader(test_ekyn_ids,test_snezana_mice_ids,courtney_ids=test_courtney_ids,batch_size=batch_size,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SleepStageClassifier()
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
criterion = nn.CrossEntropyLoss()

print(len(trainloader) * batch_size,len(testloader) * batch_size)
print(device)

validation_frequency_epochs = 1
best_dev_loss = torch.inf
best_dev_loss_epoch = 0
best_dev_f1 = 0
best_dev_f1_epoch = 0
ma_window_size = 10

model.to(device)
model.train()

trainlossi = []
testlossi = []
train_f1s = []
test_f1s = []

for epoch in range(1000):
    epoch_train_f1s = []  # Collect F1s for each batch in this epoch
    epoch_train_losses = []  # Collect losses for each batch in this epoch
    
    for Xi, yi in tqdm(trainloader):
        Xi, yi = Xi.to(device), yi.to(device)    
        logits = model(Xi)
        optimizer.zero_grad()
        loss = criterion(logits, yi)
        loss.backward()
        optimizer.step()
        
        # Calculate and store F1 for this batch
        batch_f1 = calculate_f1(logits, yi)
        epoch_train_f1s.append(batch_f1)
        epoch_train_losses.append(loss.item())
    
    # Add average loss and F1 for this epoch
    trainlossi.extend(epoch_train_losses)
    train_f1s.append(np.mean(epoch_train_f1s))
    
    if epoch % validation_frequency_epochs == 0:
        model.eval()
        all_test_preds = []
        all_test_labels = []
        test_losses = []

        with torch.no_grad():
            for Xi, yi in testloader:
                Xi, yi = Xi.to(device), yi.to(device)
                logits = model(Xi)
                loss = criterion(logits, yi)
                test_losses.append(loss.item())
                all_test_preds.append(logits.argmax(dim=1))
                all_test_labels.append(yi.argmax(dim=1))
            all_test_preds = torch.cat(all_test_preds).cpu()
            all_test_labels = torch.cat(all_test_labels).cpu()
            avg_test_loss = np.mean(test_losses)
            test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')

            testlossi.append(avg_test_loss)
            test_f1s.append(test_f1)

            # Track best model by loss
            if avg_test_loss < best_dev_loss:
                best_dev_loss = avg_test_loss
                best_dev_loss_epoch = epoch
                torch.save(model.state_dict(), '../models/best_model_by_loss.pt')
                
            # Track best model by F1
            if test_f1 > best_dev_f1:
                best_dev_f1 = test_f1
                best_dev_f1_epoch = epoch
                torch.save(model.state_dict(), '../models/best_model_by_f1.pt')
                
            # Call the updated plotting function with both loss and F1 data
            plot_training_progress(
                trainlossi,
                testlossi,
                train_f1s,
                test_f1s,
                ma_window_size,
                '../models/training_metrics.jpg'
            )
            print(f"Epoch {epoch}: Train Loss: {np.mean(epoch_train_losses):.4f}, Test Loss: {avg_test_loss:.4f}")
            print(f"Train F1: {train_f1s[-1]:.4f}, Test F1: {test_f1:.4f}")
            print(f"Best Test Loss: {best_dev_loss:.4f} (Epoch {best_dev_loss_epoch})")
            print(f"Best Test F1: {best_dev_f1:.4f} (Epoch {best_dev_f1_epoch})")


y = torch.vstack([torch.vstack([model(Xi.to(device)).softmax(dim=1).argmax(dim=1).detach().cpu(),yi.argmax(dim=1).detach().cpu()]).T for Xi,yi in trainloader])
y_pred = y[:,0]
y_true = y[:,1]
print(classification_report(y_true=y_true,y_pred=y_pred))
ConfusionMatrixDisplay.from_predictions(y_true,y_pred)

y = torch.vstack([torch.vstack([model(Xi.to(device)).softmax(dim=1).argmax(dim=1).detach().cpu(),yi.argmax(dim=1).detach().cpu()]).T for Xi,yi in testloader])
y_pred = y[:,0]
y_true = y[:,1]
print(classification_report(y_true=y_true,y_pred=y_pred))
ConfusionMatrixDisplay.from_predictions(y_true,y_pred)