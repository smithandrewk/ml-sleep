from ekyn import *
import json
import argparse
import inspect
import ekyn

path_to_experiments = f'../experiments'

# Function to get all subclasses of nn.Module from a module
def get_model_classes(module):
    return [cls for name, cls in inspect.getmembers(module, inspect.isclass) 
            if issubclass(cls, nn.Module) and cls != nn.Module]
# Get model classes dynamically from ekyn module
model_classes = get_model_classes(ekyn)
print(model_classes)

parser = argparse.ArgumentParser(description='Training program')
parser.add_argument("--device", type=int, default=0,help="Cuda Device")
parser.add_argument("--batch", type=int, default=512,help="Batch Size")
parser.add_argument("--lr", type=float, default=3e-4,help="Learning Rate")
parser.add_argument("--dataset", type=str, default=f'../pt_ekyn_robust_50hz',help="Path To Dataset")
parser.add_argument("--model", type=int, default=None, help="Model to use (enter number)", action='store')
args = parser.parse_args()

# If no model was specified, list available models and exit
if args.model is None:
    print("Available models:")
    for index, model_class in enumerate(model_classes):
        print(f"{index}: {model_class.__name__}")
    parser.print_help()
    exit()

# Check if the model number is valid
if args.model < 0 or args.model >= len(model_classes):
    print(f"Error: Invalid model number. Choose from 0 to {len(model_classes) - 1}")
    exit()

# Instantiate the model by indexing into the list
model = model_classes[args.model]()

ids = sorted(set([filename.split('_')[0] for filename in os.listdir(args.dataset)]))
test_ids = [ids.pop(0),ids.pop(0),ids.pop(0),ids.pop(0)]
train_ids = ids[:4]

config = {
    'device':f'cuda:{args.device}',
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
    'testf1w':[],
    'lr':args.lr,
    'weight_decay':1e-2,
    'batch_size':args.batch,
    'model': model_classes[args.model].__name__,
    'dataset':args.dataset
}

print("using dataset",config["dataset"])
if not os.path.exists(path_to_experiments):
    os.makedirs(path_to_experiments)
if len(os.listdir(path_to_experiments)) == 0:
    experiment_id = 0
else:
    experiment_id = len(os.listdir(path_to_experiments))
os.makedirs(f'{path_to_experiments}/{experiment_id}',exist_ok=True)

trainloader,unweighted_trainloader,unweighted_testloader = get_dataloaders(train_ids,test_ids,batch_size=config["batch_size"],path_to_dataset=config["dataset"])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
best_dev_loss = float('inf')
best_dev_loss_epoch = 0
best_dev_f1 = 0
best_dev_f1_epoch = 0
window_size = 10

model.to(config["device"])

validation_frequency_epochs = 5
loss_offset = 2

for epoch in tqdm(range(10000)):
    for Xi, yi in trainloader:
        Xi, yi = Xi.to(config["device"]), yi.to(config["device"])
        logits = model(Xi)
        loss = criterion(logits, yi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % validation_frequency_epochs == 0:
        loss, f1, report = evaluate(unweighted_trainloader, model, criterion, config["device"])
        config["trainlossi"].append(loss)
        config["trainf1i"].append(report['macro avg']['f1-score'])
        config["trainf1p"].append(report['0']['f1-score'])
        config["trainf1s"].append(report['1']['f1-score'])
        config["trainf1w"].append(report['2']['f1-score'])
        
        loss, f1, report = evaluate(unweighted_testloader, model, criterion, config["device"])
        config["testlossi"].append(loss)
        config["testf1i"].append(report['macro avg']['f1-score'])
        config["testf1p"].append(report['0']['f1-score'])
        config["testf1s"].append(report['1']['f1-score'])
        config["testf1w"].append(report['2']['f1-score'])

        # Update best models
        if config["testlossi"][-1] < best_dev_loss:
            torch.save(model.state_dict(), f'{path_to_experiments}/{experiment_id}/model_bestdevloss.pt')
            best_dev_loss = config["testlossi"][-1]
            best_dev_loss_epoch = epoch // validation_frequency_epochs
        if config["testf1i"][-1] > best_dev_f1:
            torch.save(model.state_dict(), f'{path_to_experiments}/{experiment_id}/model_bestdevf1.pt')
            best_dev_f1 = config["testf1i"][-1]
            best_dev_f1_epoch = epoch // validation_frequency_epochs

        update_plot(epoch, config["trainlossi"][loss_offset:], config["testlossi"][loss_offset:], config["trainf1i"][loss_offset:], config["testf1i"][loss_offset:], config["trainf1p"][loss_offset:], config["trainf1s"][loss_offset:], config["trainf1w"][loss_offset:], config["testf1p"][loss_offset:], config["testf1s"][loss_offset:], config["testf1w"][loss_offset:], best_dev_loss, best_dev_loss_epoch-loss_offset, best_dev_f1, best_dev_f1_epoch-loss_offset, window_size)
        plt.savefig(f'{path_to_experiments}/{experiment_id}/loss_with_ma.jpg')
        plt.savefig(f'loss.jpg')
        plt.close()
        torch.save(model.state_dict(), f'{path_to_experiments}/{experiment_id}/last_model.pt')
        with open(f'{path_to_experiments}/{experiment_id}/config.json','w') as f:
            json.dump(config,f)
