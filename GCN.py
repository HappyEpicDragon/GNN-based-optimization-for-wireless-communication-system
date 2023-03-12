from model import *

LEARNING_RATE = 0.1
WEIGHT_DACAY = 5e-4
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataset = CoraData().data
node_feature = dataset.x / dataset.x.sum(1, keepdims=True)
tensor_x = tensor_from_numpy(node_feature, DEVICE)
tensor_y = tensor_from_numpy(dataset.y, DEVICE)
tensor_train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)
tensor_val_mask = tensor_from_numpy(dataset.val_mask, DEVICE)
tensor_test_mask = tensor_from_numpy(dataset.test_mask, DEVICE)
normalize_adjacency = CoraData.normalization(dataset.adjacency)

num_nodes, input_dim = node_feature.shape

indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
                                       normalize_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes)).to(DEVICE)
model = GcnNet(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DACAY)


def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(EPOCHS):
        logits = model(tensor_adjacency, tensor_x)
        train_mask_logits = logits[tensor_train_mask]
        loss = criterion(train_mask_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, _, _ = test(tensor_train_mask)
        val_acc, _, _ = test(tensor_val_mask)
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, val_acc_history


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
        return accuracy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()

def GCN_main():
    loss, val_acc = train()  # 每个epoch 模型在训练集上的loss 和验证集上的准确率
    # 计算最后训练好的模型在测试集上准确率
    test_acc, test_logits, test_label = test(tensor_test_mask)
    print("Test accuarcy: ", test_acc.item())
    plot_loss_with_acc(loss, val_acc)
