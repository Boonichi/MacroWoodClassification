import torch
import torchvision

def train_one_epoch(model, criterion, train_dataloader, optimizer, epochs, device = "cpu", log_interval = 100):
    model.train(True)
    
    log_interval = 10

    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader, 0):
            samples, targets = data

            # zero the parameter gradients
            optimizer.zero_grad()

            samples.to(device)
            targets.to(device)

            # forward + backward + optimize
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == log_interval - 1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    
            
