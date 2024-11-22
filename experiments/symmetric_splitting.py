"""This script runs symmetric split HMC on LeNet for FashionMNIST."""
# %%
# Note that you have to install the following packages to run this script separately
# as they are not and should not be part of the core source package
import hamiltorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(1)

# get fashion mnist data
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(
    root='../../data', train=True, download=True, transform=transform
)
# take only 90% of the training data and set apart 10% for validation for fair comp
torch.manual_seed(0)
train_data, val_data = torch.utils.data.random_split(train_data, [54000, 6000])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.FashionMNIST(
    root='../../data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# %%
class LeNet(nn.Module):
    """LeNet with 2 convolutional layers and 3 fully connected layers."""

    def __init__(self):
        """Initialize the LeNet model."""
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(
            84, 10
        )  # Assuming the output dimension is 10 for FashionMNIS

    def forward(self, x):
        """Forward pass of the LeNet model."""
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = LeNet()

# set hyperparameters
step_size = 0.0005
num_samples = 3300
L = 30
burn = 299
store_on_GPU = False
debug = False
model_loss = 'multi_class_linear_output'
mass = 0.01  # values taken from the paper
tau = 1.0  # Prior Precision


tau_list = []
for w in net.parameters():
    tau_list.append(
        tau
    )  # set the prior precision to be the same for each set of weights
tau_list = torch.tensor(tau_list).to(device)

# Set initial weights
params_init = hamiltorch.util.flatten(net).to(device).clone()
# Set the Inverse of the Mass matrix
inv_mass = torch.ones(params_init.shape) / mass

# length of data // batch size
M = len(train_data) // 64

print(params_init.shape)
integrator = hamiltorch.Integrator.SPLITTING
sampler = hamiltorch.Sampler.HMC

params_hmc_s = hamiltorch.samplers.sample_split_model(
    net,
    train_loader,
    params_init=params_init,
    num_splits=M,
    num_samples=num_samples,
    step_size=step_size,
    num_steps_per_sample=L,
    inv_mass=inv_mass.to(device),
    integrator=integrator,
    debug=debug,
    store_on_GPU=store_on_GPU,
    burn=burn,
    sampler=sampler,
    tau_list=tau_list,
    model_loss=model_loss,
)

params_hmc_gpu = [ll.to(device) for ll in params_hmc_s[1:]]

pred_list_split, _ = hamiltorch.predict_model(
    net,
    test_loader=test_loader,
    samples=params_hmc_gpu,
    model_loss=model_loss,
    tau_list=tau_list,
)

# %%
# move everything to the CPU to avoid memory issues
pred_list_split = pred_list_split.cpu()
test_data.targets = test_data.targets.cpu()

pred = torch.max(pred_list_split, 2)[1]
# compute majority vote over first dimension not just the mean!
majority_vote = torch.mode(pred, 0)[0]
correct = (majority_vote.float() == test_data.targets).sum()
accuracy = correct / len(test_data)
print(f'Accuracy: {accuracy}')

# %%
# calculate the lppd using log probs from the categorical distribution
# first calculate the log likelihood for each sample pred_list_split has shape
# (num_samples, batch_size, num_classes)
log_likelihoods = torch.distributions.Categorical(logits=pred_list_split).log_prob(
    test_data.targets
)
# exponentiate to get likelihoods and average over the sample dimension
likelihoods = torch.exp(log_likelihoods).mean(0)
# calculate the lppd
lppd = torch.log(likelihoods).mean()
print(f'LPPD: {lppd}')
