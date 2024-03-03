import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import time
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, num_epochs=25, inception=False, debug=False, output=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_values = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs+1):
        if epoch % 5 == 0 and debug:
            print(f'Epoch {epoch}/{num_epochs}')
            
    
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if inception:
                        _, preds = torch.max(outputs, 1)
                    else:
                        _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(dataloaders[phase])
            loss_values.append(epoch_loss)
            epoch_acc =  running_corrects / total
            if epoch % 5 == 0 and debug:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')   
                

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 and debug:  
            print('-' * 10)

    time_elapsed = time.time() - since
    if output:
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        print(f'Best val loss: {best_val_loss:4f}')


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_values