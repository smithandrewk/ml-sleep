from ekyn import *
import json

experiments_dir = 'data'
ids = sorted(set([filename.split('_')[0] for filename in os.listdir(f'pt_ekyn_robust_50hz')]))
test_ids = [ids.pop(1)]
train_ids = ids

config = {
    'device':'cuda',
    'train_ids':train_ids,
    'test_ids':test_ids,
    'trainlossi':[],
    'testlossi':[],
    'trainf1i':[],
    'testf1i':[],
    'trainf1p':[],
    'testf1p':[],
    'trainf1s':[],
    'testf1s':[],
    'trainf1w':[],
    'testf1w':[]
}

if len(os.listdir(experiments_dir)) == 0:
    experiment_id = 0
else:
    experiment_id = len(os.listdir(experiments_dir))
os.makedirs(f'{experiments_dir}/{experiment_id}',exist_ok=True)

trainloader,unweighted_trainloader,unweighted_testloader = get_dataloaders(train_ids,test_ids,batch_size=4196,num_workers=2)
model = SleepStageClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(),lr=3e-4,weight_decay=1e-2)
best_dev_loss = float('inf')
best_dev_loss_epoch = 0
best_dev_f1 = 0
best_dev_f1_epoch = 0
window_size = 10

model.to(config["device"])

validation_frequency_epochs = 5
loss_offset = 5

for epoch in tqdm(range(10000)):
    for Xi, yi in trainloader:
        Xi, yi = Xi.to(config["device"]), yi.to(config["device"])
        logits = model(Xi)
        loss = criterion(logits, yi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # if epoch % validation_frequency_epochs == 0:
    #     loss, f1, report = evaluate(unweighted_trainloader, model, criterion, config["device"])
    #     config["trainlossi"].append(loss)
    #     config["trainf1i"].append(report['macro avg']['f1-score'])
    #     config["trainf1p"].append(report['0']['f1-score'])
    #     config["trainf1s"].append(report['1']['f1-score'])
    #     config["trainf1w"].append(report['2']['f1-score'])
        
    #     loss, f1, report = evaluate(unweighted_testloader, model, criterion, config["device"])
    #     config["testlossi"].append(loss)
    #     config["testf1i"].append(report['macro avg']['f1-score'])
    #     config["testf1p"].append(report['0']['f1-score'])
    #     config["testf1s"].append(report['1']['f1-score'])
    #     config["testf1w"].append(report['2']['f1-score'])

    #     # Update best models
    #     if config["testlossi"][-1] < best_dev_loss:
    #         # torch.save(model.state_dict(), 'model_bestdevloss.pt')
    #         best_dev_loss = config["testlossi"][-1]
    #         best_dev_loss_epoch = epoch // validation_frequency_epochs
    #     if config["testf1i"][-1] > best_dev_f1:
    #         # torch.save(model.state_dict(), 'model_bestdevf1.pt')
    #         best_dev_f1 = config["testf1i"][-1]
    #         best_dev_f1_epoch = epoch // validation_frequency_epochs

    #     update_plot(epoch, config["trainlossi"][loss_offset:], config["testlossi"][loss_offset:], config["trainf1i"][loss_offset:], config["testf1i"][loss_offset:], config["trainf1p"][loss_offset:], config["trainf1s"][loss_offset:], config["trainf1w"][loss_offset:], config["testf1p"][loss_offset:], config["testf1s"][loss_offset:], config["testf1w"][loss_offset:], best_dev_loss, best_dev_loss_epoch-loss_offset, best_dev_f1, best_dev_f1_epoch-loss_offset, window_size)
    #     plt.savefig(f'{experiments_dir}/{experiment_id}/loss_with_ma.jpg')
    #     plt.savefig(f'loss.jpg')
    #     plt.close()
    #     torch.save(model.state_dict(), f'{experiments_dir}/{experiment_id}/last_model.pt')
    #     with open(f'{experiments_dir}/{experiment_id}/config.json','w') as f:
    #         json.dump(config,f)