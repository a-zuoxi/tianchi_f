import torch
device = 'cpu'
pred = torch.tensor([-1,0,1], dtype=torch.float64, device=device, requires_grad=True)
y = torch.tensor([2,0,3], dtype=torch.float64, device=device)
loss = torch.nn.functional.l1_loss(pred, y, reduction='sum')
loss.backward()
print(pred.grad)
