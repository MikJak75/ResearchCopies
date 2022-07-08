def split_input_arrays(in_data,label_data,size_split):

    xtr, xjunk, ytr, yjunk = train_test_split(in_data,label_data,train_size=size_split)
    xv, xte, yv, yte = train_test_split(xjunk,yjunk, test_size=0.5)
    print('xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape')
    print(xtr.shape, ytr.shape, xv.shape, yv.shape, xte.shape, yte.shape)
    return xtr, ytr, xv, yv, xte, yte

def get_dataloaders_fromsplitarrays(xtr,ytr,xv,yv,xte,yte,batch_size):

    tr_set = torch.utils.data.TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float())
    tr_load = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True)

    va_set = torch.utils.data.TensorDataset(torch.from_numpy(xv).float(), torch.from_numpy(yv).float())
    va_load = torch.utils.data.DataLoader(va_set, batch_size=batch_size, shuffle=True)

    te_set = torch.utils.data.TensorDataset(torch.from_numpy(xte).float(), torch.from_numpy(yte).float())
    te_load = torch.utils.data.DataLoader(te_set, batch_size=batch_size, shuffle=True)

    return tr_set, va_set, te_set, tr_load, va_load, te_load 

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []
    for data,label in dataloader: 
        img = data
        img = img.view(img.size(0), -1).to(device)  
        label = label.to(device)
        latent = encoder(img)
        decoded_img = decoder(latent)
        loss = loss_fn(decoded_img, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def valid_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): 
        list_decoded_img = []
        list_img = []
        for  data, label in dataloader:
            img = data
            img = img.view(img.size(0), -1).to(device) 
            label = label.to(device)
            latent = encoder(img)
            decoded_img = decoder(latent)
            list_decoded_img.append(decoded_img.cpu())
            list_img.append(img.cpu())
        list_decoded_img = torch.cat(list_decoded_img)
        list_img = torch.cat(list_img) 
        val_loss = loss_fn(list_decoded_img, list_img)
    return val_loss.data

def plot_ae_outputs(encoder,decoder,dataset,device,n=10):
    plt.figure(figsize=(26,5.5))
    for i in range(10):
      ax = plt.subplot(2,n,i+1)
      img,_ = dataset[i]
      #Notice that below i'm loading an image only, so it needs to be flatten
      #before entering the network
      img = torch.flatten(img).to(device)
      encoder.eval().to(device)
      decoder.eval().to(device)
      with torch.no_grad():
        decoded_img = decoder(encoder(img))
      plt.plot(img.cpu().reshape(2,128).numpy()[0],img.cpu().reshape(2,128).numpy()[1]) 
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n) 
      plt.plot(decoded_img.cpu().reshape(2,128).numpy()[0],decoded_img.cpu().reshape(2,128).numpy()[1]) 
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show() 
