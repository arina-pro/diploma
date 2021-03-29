import numpy as np
from scipy import linalg
from scipy.sparse import isspmatrix_coo, isspmatrix, coo_matrix
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

n_samples = 40
n_features = 20
n_classes = 2
torch.manual_seed(12345)
prng = np.random.RandomState(1)

### From sklearn's Sparse inverse covariance estimation
# data generation

# first the precision matrix
prec = make_sparse_spd_matrix(n_features, alpha=.98,
                              smallest_coef=.4,
                              largest_coef=.7,
                              random_state=prng)
cov = linalg.inv(prec) # the covariance matrix, the inverse of precision
d = np.sqrt(np.diag(cov)) # this is sqrt of the diagonal values of the inverse covariance?
cov /= d # then we divide?
cov /= d[:, np.newaxis] # again, differently?
prec *= d # this, we multiply?
prec *= d[:, np.newaxis] # and again
X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples) # draw samples from the higher-dimensional Gaussian
# Standardize
X -= X.mean(axis=0)
X /= X.std(axis=0)
print(isspmatrix(X))
print(isspmatrix_coo(X))
### --

def create_data(n_features, n_classes, x, edge_index):
    data = Data(x=x, edge_index=edge_index)
    data.train_mask = None
    data.val_mask = None
    data.test_mask = None
    return data


x = None
edge_index = None
data = create_data(n_features, n_classes, x, edge_index)
train_data, test_data = train_test_split(data, test_size=0.33) # Or do it through masks?
data_list = [create_data(n_features, n_classes, x, edge_index) for _ in range(n_samples)]
loader = DataLoader(data_list, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(n_features, 16, cached=True)
        self.conv2 = GCNConv(16, n_classes, cached=True)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution. <- Need to see if it makes sense
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
    scheduler.step()