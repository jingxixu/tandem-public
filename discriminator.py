import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import misc_utils as mu
import copy
from discriminator_dataset import VariedMNISTDataset
import tqdm
import icp
import os


class Discriminator:
    def predict(self, obs):
        """
        obs is of shape (1, height, width) or (n, 1, height, weight).
        obs is of numpy array. obs is the raw obs from the env, which includes the agent information and grey pixels
        return
            prediction, int or (n, ) np array
            max_prob, int or (n, ) np array
            probs, (n, ) or (n, num_classes) np array
        """
        raise NotImplementedError


class DummyDiscriminator(Discriminator):
    """ this predicts 0 for anything and gives equal probs to all classes"""
    pattern = 'dummy'

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def predict(self, obs):
        prediction = 0
        probs = np.array([1 / self.num_classes] * self.num_classes)
        max_prob = 1 / self.num_classes

        if obs.ndim == 4:
            num_envs = obs.shape[0]
            return np.array([prediction] * num_envs), np.array([max_prob] * num_envs), np.tile(probs, (num_envs, 1))
        elif obs.ndim == 3:
            return prediction, max_prob, probs


class GroundTruthDiscriminator:
    pattern = 'gt'

    def __init__(self, gt_path=None):
        self.grids = np.load(gt_path) if gt_path is not None else None

    def set_grids(self, grids):
        self.grids = grids

    def set_gt_path(self, gt_path):
        self.grids = np.load(gt_path)

    def predict(self, obs):
        discriminator_input = mu.get_discriminator_input(obs)
        if obs.ndim == 4:
            prediction = []
            max_probs = []
            probs = []
            for i in range(discriminator_input.shape[0]):
                p, m_prob, ps = mu.check_class(discriminator_input[i], self.grids)
                prediction.append(p)
                max_probs.append(m_prob)
                probs.append(ps)
            return np.array(prediction), np.array(max_probs), np.array(probs)
        elif obs.ndim == 3:
            return mu.check_class(discriminator_input, self.grids)


class LearnedDiscriminator:
    pattern = 'learned'

    def __init__(self,
                 height,
                 width,
                 lr=0.001,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 model_path=None,
                 gamma=0.7,
                 save_dir=None):
        self.device = device
        self.model_path = model_path
        self.model = DiscriminatorNet(self.device, height=height, width=width)
        self.lr = lr
        self.gamma = gamma
        self.save_dir = save_dir
        if self.model_path is not None:
            self.load_model(model_path=self.model_path)
        self.model.to(self.device)
        self.optimizer = None
        self.loss = 0

    def learn(self, epochs, train_loader, test_loader, use_best_model=True, logger=None):
        log = logger.log if logger is not None else print
        # the optimizer stats (such as moving averages for ADAM) should be reset
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)

        # stats
        stats = []
        # the stat for the best model
        model_path = None
        test_acc = None
        test_loss = None
        train_loss = None
        train_acc = None
        max_test_acc = 0

        for i in range(epochs):
            train_loss, train_acc = self.train_epoch(i, train_loader, logger)
            test_loss, test_acc = self.test_epoch(i, test_loader, logger)
            # self.scheduler.step()

            stats.append({
                'epoch': i,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            })

            # save dir is changing with different batch of data collected for discriminator
            if self.save_dir is not None and test_acc >= max_test_acc:
                # for each epoch
                model_folder_name = f'epoch_{i:06d}_loss_{test_loss:.6f}_acc_{test_acc:.8f}'
                if not os.path.exists(os.path.join(self.save_dir, model_folder_name)):
                    os.makedirs(os.path.join(self.save_dir, model_folder_name))
                model_path = os.path.join(self.save_dir, model_folder_name, 'model.pth')
                torch.save(self.model.state_dict(), model_path)
                log(f'model saved to {model_path}\n')
                stats[i]['model_path'] = model_path
                max_test_acc = test_acc

        if use_best_model and self.save_dir is not None:
            # return the best model according to the testing accuracy, this will pick the later one with equal acc
            best_stat = sorted(stats, key=lambda x: x['test_acc'])[-1]
            model_path, test_acc, test_loss, train_loss, train_acc = \
                best_stat['model_path'], best_stat['test_acc'], best_stat['test_loss'], best_stat['train_loss'], best_stat['train_acc']
            self.load_model(model_path=model_path)
            log(f're-loaded model path {model_path}')
        return model_path, train_loss, train_acc, test_loss, test_acc, stats

    def train_epoch(self, epoch, data_loader, logger=None):
        log = logger.log if logger is not None else print
        pbar = tqdm.tqdm(total=len(data_loader.dataset))
        self.model.train()
        correct = 0
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = data.float()
            self.optimizer.zero_grad()
            output = self.model.forward_logprob(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            # item() is important here for saving memory
            # https://discuss.pytorch.org/t/cpu-ram-usage-increasing-for-every-epoch/24475/6
            epoch_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update(data.shape[0])

        pbar.close()
        epoch_loss = epoch_loss / len(data_loader.dataset)
        acc = correct / len(data_loader.dataset)
        log('Train Epoch: {} | Loss: {:.6f} | Acc: {:.6f}'.format(epoch, epoch_loss, acc))

        return epoch_loss, acc

    def test_epoch(self, epoch, data_loader, logger=None):
        log = logger.log if logger is not None else print
        pbar = tqdm.tqdm(total=len(data_loader.dataset))
        self.model.eval()
        correct = 0
        epoch_loss = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.float()
                output = self.model.forward_logprob(data)
                epoch_loss += F.nll_loss(output, target, reduction='sum')
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.update(data.shape[0])

        pbar.close()
        epoch_loss = epoch_loss.item() / len(data_loader.dataset)
        acc = correct / len(data_loader.dataset)
        log('Test Epoch: {} | Loss: {:.6f} | Acc: {:.6f}\n'.format(epoch, epoch_loss, acc))
        self.loss = epoch_loss
        return epoch_loss, acc

    def predict(self, obs):
        obs = mu.get_discriminator_input(obs)
        # this ob comes from the env.step and it needs to be normalized
        obs = (obs / 255.0 - 0.5) / 0.5
        return self.model.predict(obs)

    def predict_with_representation(self, obs):
        obs = mu.get_discriminator_input(obs)
        # this ob comes from the env.step and it needs to be normalized
        obs = (obs / 255.0 - 0.5) / 0.5
        return self.model.predict_with_representation(obs)

    def save_model(self, model_dir, model_name):
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f'model loaded from {model_path}')


class EnsembleDiscriminator(Discriminator):
    pattern = 'ensemble'

    def __init__(self,
                 num_models,
                 height,
                 width,
                 lr=0.001,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 model_path=None,
                 gamma=0.7,
                 save_dir=None):
        # TODO model path
        self.num_models = num_models
        self.device = device
        self.model_path = model_path
        self.save_dir = save_dir
        # called models but actually discriminators
        self.models = [LearnedDiscriminator(height=height,
                                            width=width,
                                            lr=lr,
                                            device=device,
                                            model_path=None,
                                            gamma=gamma,
                                            save_dir=None) for _ in range(self.num_models)]
        if self.model_path is not None:
            model_name_list = os.listdir(self.model_path)
            for i in range(self.num_models):
                self.models[i].load_model(os.path.join(self.model_path, model_name_list[i]))

    def learn(self, epochs, train_loader, test_loader, logger=None):
        log = logger.log if logger is not None else print
        for i in range(self.num_models):
            log(f'\nlearning model {i}\n')
            _, stats = self.models[i].learn(epochs=epochs, train_loader=train_loader, test_loader=test_loader, use_best_model=False, logger=logger)

        if self.save_dir is not None:
            model_folder_name = f'model'
            if not os.path.exists(os.path.join(self.save_dir, model_folder_name)):
                os.makedirs(os.path.join(self.save_dir, model_folder_name))
            for i in range(self.num_models):
                model_path = os.path.join(self.save_dir, model_folder_name, f'model_{i}.pth')
                torch.save(self.models[i].model.state_dict(), model_path)
                log(f'model saved to {model_path}\n')
            model_path = os.path.join(self.save_dir, model_folder_name)

        # the model path is the path to the folder containing all ensemble models; stats is the stats for the last model
        return model_path, stats

    def predict(self, obs):
        prediction_list, max_prob_list, probs_list = [], [], []
        for i in range(self.num_models):
            prediction, max_prob, probs = self.models[i].predict(obs)
            prediction_list.append(prediction)
            max_prob_list.append(max_prob)
            probs_list.append(probs)

        # disagreement based
        # probs = np.zeros(10)
        # for p, p_c in zip(*np.unique(prediction_list, return_counts=True)):
        #     probs[p] = p_c / self.num_models
        # max_prob = np.max(probs)
        # prediction = np.argmax(probs)

        # single model
        # probs = probs_list[0]
        # prediction = prediction_list[0]
        # max_prob = max_prob_list[0]

        if obs.ndim == 3:
            probs = np.average(np.array(probs_list), axis=0)
            max_prob = np.max(probs)
            prediction = np.argmax(probs)
        elif obs.ndim == 4:
            # probs_list is a list of (n_envs, n_classes) np array
            probs = np.average(np.array(probs_list), axis=0)
            max_prob = np.max(probs, axis=1)
            prediction = np.argmax(probs, axis=1)
        else:
            raise ValueError
        return prediction, max_prob, probs


class DiscriminatorNet(nn.Module):
    """
    model and input need to be on the same device
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, device, height, width):
        super(DiscriminatorNet, self).__init__()
        self.height = height
        self.width = width

        if self.width == self.height == 50:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(33856, 128)
            self.fc2 = nn.Linear(128, 10)
        elif self.width == self.height == 28:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
        elif self.width == self.height == 60:
            # architecture (3)
            # a much bigger one, without dropout
            # # n, 1, 60, 60
            # self.conv1 = nn.Conv2d(1, 64, 3, 1)
            # # n, 64, 58, 58
            # self.conv2 = nn.Conv2d(64, 128, 3, 2)
            # # n, 128, 28, 28
            # self.conv3 = nn.Conv2d(128, 256, 3, 2)
            # # n, 256. 13, 13
            # self.conv4 = nn.Conv2d(256, 256, 3, 2)
            # # n, 256, 6, 6
            # self.conv5 = nn.Conv2d(256, 256, 3, 2)
            # # n, 256, 2, 2
            # self.fc1 = nn.Linear(1024, 512)
            # # n, 512
            # self.fc2 = nn.Linear(512, 512)
            # # n, 512
            # self.fc3 = nn.Linear(512, 10)
            # # n, 10

            # architecture (1)
            # this is from before but with first stride 1 layer and dropout
            # # n, 1, 60, 60
            # self.conv1 = nn.Conv2d(1, 32, 3, 1)
            # # n, 32. 58, 58
            # self.conv2 = nn.Conv2d(32, 64, 3, 2)
            # # n, 64, 28, 28
            # self.conv3 = nn.Conv2d(64, 64, 3, 2)
            # # n, 64, 13, 13
            # self.conv4 = nn.Conv2d(64, 64, 3, 2)
            # # n. 64. 6, 6
            # self.fc1 = nn.Linear(2304, 128)
            # # n, 128
            # self.fc2 = nn.Linear(128, 10)
            # # n, 10
            # self.dropout = nn.Dropout(0.5)

            # architecture (2)
            # this is from before
            # conv architecture, 352352 weights
            # # n, 1, 60, 60
            # self.conv1 = nn.Conv2d(1, 32, 3, 2)
            # # n, 32, 29, 29
            # self.conv2 = nn.Conv2d(32, 64, 3, 2)
            # # n, 64, 14, 14
            # self.conv3 = nn.Conv2d(64, 64, 3, 2)
            # # n. 64. 6, 6
            # self.fc1 = nn.Linear(2304, 128)
            # # n, 128
            # self.fc2 = nn.Linear(128, 10)
            # # n, 10

            # architecture (4)
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(50176, 128)
            self.fc2 = nn.Linear(128, 10)

            # architecture (5)
            # self.fc1 = nn.Linear(3600, 2048)
            # self.fc2 = nn.Linear(2048, 1024)
            # self.fc3 = nn.Linear(1024, 512)
            # self.fc4 = nn.Linear(512, 10)

        self.device = device
        self.to(device)

    def forward(self, x):
        # this function returns the logits
        if self.width == self.height == 50 or self.width == self.height == 28:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
        elif self.width == self.height == 60:
            # architecture (3)
            # x = self.conv1(x)
            # x = F.relu(x)
            # x = self.conv2(x)
            # x = F.relu(x)
            # x = self.conv3(x)
            # x = F.relu(x)
            # x = self.conv4(x)
            # x = F.relu(x)
            # x = self.conv5(x)
            # x = F.relu(x)
            # x = torch.flatten(x, 1)
            # x = self.fc1(x)
            # x = F.relu(x)
            # x = self.fc2(x)
            # x = F.relu(x)
            # x = self.fc3(x)

            # architecture (1)
            # x = self.conv1(x)
            # x = F.relu(x)
            # x = self.conv2(x)
            # x = F.relu(x)
            # x = self.conv3(x)
            # x = F.relu(x)
            # x = self.conv4(x)
            # x = F.relu(x)
            # x = torch.flatten(x, 1)
            # x = self.fc1(x)
            # x = F.relu(x)
            # x = self.dropout(x)
            # x = self.fc2(x)

            # architecture (4)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            # architecture (2)
            # x = self.conv1(x)
            # x = F.relu(x)
            # x = self.conv2(x)
            # x = F.relu(x)
            # x = self.conv3(x)
            # x = F.relu(x)
            # x = torch.flatten(x, 1)
            # x = self.fc1(x)
            # x = F.relu(x)
            # x = self.fc2(x)

            # architecture (5)
            # x = torch.flatten(x, 1)
            # x = self.fc1(x)
            # x = F.relu(x)
            # x = self.fc2(x)
            # x = F.relu(x)
            # x = self.fc3(x)
            # x = F.relu(x)
            # x = self.fc4(x)

        else:
            raise TypeError('wrong image size')
        return x

    def forward_with_representation(self, x):
        # same as self.forward but return also the intermediate representation
        # architecture (4)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        representation = torch.flatten(x, 1)
        x = self.fc1(representation)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x, representation

    def forward_logprob(self, x):
        x = self.forward(x)
        output = F.log_softmax(x, dim=1)
        return output

    def forward_prob(self, x):
        x = self.forward(x)
        probabilities = F.softmax(x, dim=1)
        return probabilities

    def forward_prob_with_representation(self, x):
        # same as self.forward_prob but return also the intermediate representation
        x, representation = self.forward_with_representation(x)
        probabilities = F.softmax(x, dim=1)
        return probabilities, representation

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if x.ndim == 4:
                input_x = torch.tensor(x).float()
                input_x = input_x.to(self.device)
                probs = self.forward_prob(input_x)
                probs = probs.cpu().numpy()
                max_prob = np.max(probs, axis=1)
                prediction = np.argmax(probs, axis=1)
            elif x.ndim == 3:
                input_x = torch.tensor(x[None, ...]).float()
                input_x = input_x.to(self.device)
                probs = self.forward_prob(input_x)
                probs = probs.cpu().numpy().squeeze()
                max_prob = np.max(probs)
                prediction = np.argmax(probs)
        return prediction, max_prob, probs

    def predict_with_representation(self, x):
        self.eval()
        with torch.no_grad():
            if x.ndim == 4:
                input_x = torch.tensor(x).float()
                input_x = input_x.to(self.device)
                probs, representation = self.forward_prob_with_representation(input_x)
                probs = probs.cpu().numpy()
                max_prob = np.max(probs, axis=1)
                prediction = np.argmax(probs, axis=1)
            elif x.ndim == 3:
                input_x = torch.tensor(x[None, ...]).float()
                input_x = input_x.to(self.device)
                probs, representation = self.forward_prob_with_representation(input_x)
                probs = probs.cpu().numpy().squeeze()
                max_prob = np.max(probs)
                prediction = np.argmax(probs)
        return prediction, max_prob, probs, representation


class BinaryDiscriminatorNet(nn.Module):
    """
    model and input need to be on the same device
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, device, height, width):
        super(BinaryDiscriminatorNet, self).__init__()
        self.height = height
        self.width = width

        if self.width == self.height == 50:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(33856, 128)
            self.fc2 = nn.Linear(128, 1)
        elif self.width == self.height == 28:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 1)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def forward_prob(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        probabilities = F.softmax(x, dim=1)
        return probabilities

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if x.ndim == 4:
                input_x = torch.tensor(x).float()
                input_x = input_x.to(self.device)
                probs = self.forward_prob(input_x)
                probs = probs.cpu().numpy()
                max_prob = np.max(probs, axis=1)
                prediction = np.argmax(probs, axis=1)
            elif x.ndim == 3:
                input_x = torch.tensor(x[None, ...]).float()
                input_x = input_x.to(self.device)
                probs = self.forward_prob(input_x)
                probs = probs.cpu().numpy().squeeze()
                max_prob = np.max(probs)
                prediction = np.argmax(probs)
        return prediction, max_prob, probs


class BinaryEnsembleDiscriminator(Discriminator):
    pattern = 'binary_ensemble'

    def __init__(self,
                 num_models,
                 height,
                 width,
                 lr=0.001,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 model_path=None,
                 gamma=0.7,
                 save_dir=None):
        # TODO model path
        self.num_models = num_models
        self.device = device
        self.model_path = model_path
        self.save_dir = save_dir
        # called models but actually discriminators
        self.models = [LearnedDiscriminator(height=height,
                                            width=width,
                                            lr=lr,
                                            device=device,
                                            model_path=None,
                                            gamma=gamma,
                                            save_dir=None) for _ in range(self.num_models)]
        if self.model_path is not None:
            model_name_list = os.listdir(self.model_path)
            for i in range(self.num_models):
                self.models[i].load_model(os.path.join(self.model_path, model_name_list[i]))

    def learn(self, epochs, train_loader, test_loader, logger=None):
        log = logger.log if logger is not None else print
        for i in range(self.num_models):
            log(f'\nlearning model {i}\n')
            _, stats = self.models[i].learn(epochs=epochs, train_loader=train_loader, test_loader=test_loader, use_best_model=False, logger=logger)

        if self.save_dir is not None:
            model_folder_name = f'model'
            if not os.path.exists(os.path.join(self.save_dir, model_folder_name)):
                os.makedirs(os.path.join(self.save_dir, model_folder_name))
            for i in range(self.num_models):
                model_path = os.path.join(self.save_dir, model_folder_name, f'model_{i}.pth')
                torch.save(self.models[i].model.state_dict(), model_path)
                log(f'model saved to {model_path}\n')
            model_path = os.path.join(self.save_dir, model_folder_name)

        # the model path is the path to the folder containing all ensemble models; stats is the stats for the last model
        return model_path, stats

    def predict(self, obs):
        prediction_list, max_prob_list, probs_list = [], [], []
        for i in range(self.num_models):
            prediction, max_prob, probs = self.models[i].predict(obs)
            prediction_list.append(prediction)
            max_prob_list.append(max_prob)
            probs_list.append(probs)
        if obs.ndim == 3:
            probs = np.average(np.array(probs_list), axis=0)
            max_prob = np.max(probs)
            prediction = np.argmax(probs)
        elif obs.ndim == 4:
            # probs_list is a list of (n_envs, n_classes) np array
            probs = np.average(np.array(probs_list), axis=0)
            max_prob = np.max(probs, axis=1)
            prediction = np.argmax(probs, axis=1)
        else:
            raise ValueError
        return prediction, max_prob, probs


class ICPDiscriminator(Discriminator):
    pattern = 'icp'

    def __init__(self, dataset, minimum_points=5, matching_threshold=0.0025, num_ori=10):
        # to get a more accurate ICP success rate, 
        # minimum_points = 25
        # num_ori = 36
        # but this is very hard to train with PPO explorer
        super(ICPDiscriminator, self).__init__()
        self.minimum_points = minimum_points
        self.matching_threshold = matching_threshold
        self.num_ori = num_ori
        self.height = self.weight = 60
        self.point_clouds = np.load(os.path.join('assets', 'datasets', dataset, 'point_clouds', 'point_clouds.npy'), allow_pickle=True)

    def predict_single_ob(self, ob):
        errors = []
        current_pc = mu.convert_grid_2_pc(ob)
        if len(current_pc) < self.minimum_points:
            probs = [0.1] * 10
            prediction = 0
            max_prob = 0.1
            return prediction, max_prob, probs
        for pc in self.point_clouds:
            # T, distances, i = icp.icp(current_pc, pc, max_iterations=1000, tolerance=0.0000001)
            T, error, i, angle = mu.icp_with_random_init_ori(current_pc, pc, num_ori=self.num_ori)
            errors.append(error)
            # mu.visualize_icp(current_pc, pc, T)
            # print(f'error: {error}, iter: {i}, angle: {angle}')
        matching = np.array(errors) <= self.matching_threshold
        num_matches = np.count_nonzero(matching == 1)
        if num_matches == 0:
            probs = [0.1] * 10
        else:
            probs = [1 / num_matches if matching[i] else 0 for i in range(10)]
        prediction = np.argmin(errors)
        max_prob = probs[prediction]
        # print(f'probs: {probs}')
        return prediction, max_prob, probs

    def predict(self, obs):
        # the way we use ICP is similar to how we use gt discriminator. we have a threshold for matching
        # and then compute probs based on how many matches
        if obs.ndim == 3:
            return self.predict_single_ob(obs)
        else:
            prediction = np.zeros(obs.shape[0])
            max_prob = np.zeros(obs.shape[0])
            probs = np.zeros((obs.shape[0], 10))
            for i, ob in enumerate(obs):
                p, mp, ps = self.predict_single_ob(ob)
                prediction[i] = p
                max_prob[i] = mp
                probs[i] = ps
            return prediction, max_prob, probs



