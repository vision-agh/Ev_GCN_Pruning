import multiprocessing as mp

from omegaconf import OmegaConf

from data.ncars import NCars

from models.model import MyModel

from torchmetrics import Accuracy

import torch

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    cfg = OmegaConf.load('configs/ncars.yaml')

    lm = NCars(cfg)
    lm.setup()

    model = MyModel(cfg).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=2)

    for i, batch in enumerate(lm.train_dataloader()):
        x = batch['x'].cuda()
        pos = batch['pos'].cuda()
        edge_index = batch['edge_index'].cuda()
        batch_idx = batch['batch'].cuda()

        x_out = model(x, pos, edge_index, batch_idx)

        print(x_out.shape)
        print(batch['label'].shape)
        loss = criterion(x_out, batch['label'].cuda())

        y_prediction = torch.argmax(x_out, dim=-1)
        acc = accuracy(preds=y_prediction.cpu(), target=batch['label'].cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Batch {i}: Loss: {loss.item()}, Accuracy: {acc.item()}")



    print('done')

# import torch

# x = torch.randn(1024, 1).cuda()
# pos = torch.rand(1024, 3).cuda()

# edge_index = torch.randint(0, 1024, (1024, 2)).cuda()
# mask = edge_index[:, 0] != edge_index[:, 1]
# edge_index = edge_index[mask, :]    
# edge_index = torch.unique(edge_index, dim=0)
# edge_index = torch.cat((edge_index, torch.arange(1024, device=edge_index.device).unsqueeze(1).expand(-1, 2)), dim=0)

# batch = torch.randint(0, 3, (1024,), device=edge_index.device).long()


# layer = layer.cuda()
# out = layer(x, pos, edge_index)
# print(out)


# layer.calibrate()

# out = layer(x, pos, edge_index)
# print(out)

# print(layer.observer_output.scale)


# layer.quantize()

# out = layer(x, pos, edge_index)

# print(layer.observer_output.dequantize_tensor(out))



# pool = MyGraphPooling(2, max_dimension=16)


# pos *= 16 


# print(x, pos, edge_index, batch)
# pos, x, edge_index, batch = pool(x, pos, edge_index, batch)
# print(x, pos, edge_index, batch)


