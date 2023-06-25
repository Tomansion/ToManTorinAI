import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, json


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.init_model(input_size, hidden_size, output_size)

    def init_model(self, input_size, hidden_size, output_size):
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, model_name):
        model_folder_path = "./models/" + model_name + "/"

        # Create the model folder
        os.makedirs(model_folder_path, exist_ok=True)

        # Save the model
        torch.save(self.model.state_dict(), model_folder_path + "/model.pth")

        # Save a json file with the model info
        with open(model_folder_path + "/model_info.json", "w") as f:
            json.dump(self.model_info, f)

    def load(self, model_name):
        model_folder_path = "./models/" + model_name + "/"

        # Get the model_info.json file
        with open(model_folder_path + "/model_info.json", "r") as f:
            model_info = json.load(f)
        self.init_model(
            input_size=model_info["input_size"],
            hidden_size=model_info["hidden_size"],
            output_size=model_info["output_size"],
        )

        # Load the model weights
        self.load_state_dict(torch.load(model_folder_path + "/model.pth"))

        print("Model " + model_name + " loaded")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
