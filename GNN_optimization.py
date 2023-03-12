from GNN import *
from Parameters import *

LR = 5 * 1e-2
LR_DECAY = 0.98
ITERATION_TIMES = 1000
TRAIN_SAMPLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

model = System(info, wireless_system)
Y_matrix = []
for i in range(ITERATION_TIMES):
    Y_matrix.append(model.wireless_system.pilot_signal_generator())

for i in range(ITERATION_TIMES):
    Y_matrix[i] = Y_matrix[i].to(DEVICE)
model.to(DEVICE)
criterion = minus_rate_function(model.wireless_system.A, model.wireless_system.h_d, info).to(DEVICE)
optimizer = optim.Adam(model.parameters(),
                       lr=LR,
                       weight_decay=LR_DECAY)
torch.autograd.set_detect_anomaly(True)


def train():
    loss_history = []
    model.to(DEVICE)
    model.train()
    for it in range(ITERATION_TIMES):
        Y = Y_matrix[it]
        for r in range(TRAIN_SAMPLES):
            W_v = model(Y)
            loss = criterion(W_v)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 5, norm_type=2)
            optimizer.step()
            loss_history.append(loss.item())
            # print(torch.norm(a))
            # print(loss.item())
    return loss_history


def train_main():
    loss = train()
    f = open('data.txt', 'a+')
    for index in range(len(loss)):
        f.write(loss[index] + "\n")
    f.close()
