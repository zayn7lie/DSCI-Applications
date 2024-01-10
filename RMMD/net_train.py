def train(epoch, model, device, tr_loader_x, tr_loader_y, optimizer, criterion, ld=.0, BATCH_SIZE=25):
    model.train()
    cnt, sum_loss, sum_mmd, sumloss, minloss, maxloss = 0, 0, 0, 0, 100, 0
    
    iter_y = iter(tr_loader_y)
    for batch_idx, (imgs_x, targets) in enumerate(tr_loader_x):
        imgs_x, targets = imgs_x.to(device), targets.to(device)
        
        imgs_y, _ = iter_y.__next__()
        imgs_y = imgs_y.to(device)
        cnt += 1
        if cnt % len(tr_loader_y) == 0:
            iter_y = iter(tr_loader_y)

        optimizer.zero_grad()

        output, mmd_loss = model(imgs_x, imgs_y)
        loss = criterion(output, targets)
        if ld != 0:
            loss += ld * mmd_loss
        
        sumloss += loss.item()
        minloss, maxloss = min(minloss, loss), max(maxloss, loss)
        
        sum_loss += loss.item()
        sum_mmd += mmd_loss.item()

        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % (900 / BATCH_SIZE) == 0:
            print("  - [{:.0f}/{:.0f}] Loss: AVG={:.6f} MAX={:.6f} MIN={:.6f}".format((batch_idx + 1), len(tr_loader_x), sumloss / (900 / BATCH_SIZE), maxloss, minloss))
            sumloss, minloss, maxloss = 0, 100, 0

    avg_loss, avg_mmd = sum_loss * 100 / len(tr_loader_x), sum_mmd * 100 / len(tr_loader_x)
    print("- Epoch: {:.0f} Loss: {:.11f}% MMD: {:.11f}%\n".format(epoch, avg_loss, avg_mmd))

def adjust_lr(optimizer, epoch, modellr, times=100, param=0.1):
    modellrnew = modellr * param ** (epoch // times)
    print("- Epoch:", epoch + 1, "Learning Rate:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
