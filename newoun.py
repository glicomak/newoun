from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import random

class EngineConfig():
    def __init__(self, degree, decay_rate):
        self.degree = degree
        self.decay_rate = decay_rate

class Engine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = None
        self.layers = None
    
    def forward(self, x):
        return self.layers(x)
    
    def new(self, degree, decay_rate):
        self.config = EngineConfig(degree, decay_rate)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=((degree * 27) + 1), out_features=36),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=36, out_features=27)
        )

    def load(self, file_path):
        engine_data = torch.load(file_path)
        config, params = engine_data["config"], engine_data["params"]
        self.config = EngineConfig(config["degree"], config["decay_rate"])
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=((self.config.degree * 27) + 1), out_features=36),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=36, out_features=27)
        )
        self.load_state_dict(params)

    def save(self, file_path):
        engine_data = {
            "config": {
                "degree": self.config.degree,
                "decay_rate": self.config.decay_rate
            },
            "params": self.state_dict()
        }
        torch.save(engine_data, file_path)

class Trainer():
    def prep_data(self, file):
        decay_rate, degree = self.engine.config.decay_rate, self.engine.config.degree

        nouns = file.readlines()
        X, y = [], []
        for noun in nouns:
            X_vec = []
            for i in range(degree):
                for j in range(27):
                    X_vec.append(1 if j == 26 else 0)
            decay = decay_rate
            X_vec.append(1 - decay)

            y_vec = []
            for i in range(27):
                y_vec.append(0)
            
            for char in noun:
                char_code = ord(char.upper()) - ord('A') if char != "\n" else 26
                y_vec[char_code] = 1

                X.append(X_vec.copy())
                y.append(y_vec.copy())

                y_vec[char_code] = 0
                decay *= decay_rate
                X_vec[degree * 27] = 1 - decay

                for i in range(degree):
                    if i == degree - 1:
                        for j in range(27):
                            X_vec[(i * 27) + j] = 1 if j == char_code else 0
                    else:
                        for j in range(27):
                            X_vec[(i * 27) + j] = X_vec[((i + 1) * 27) + j]

        self.X_ten = torch.tensor(X).to(dtype=torch.float32)
        self.y_ten = torch.tensor(y).to(dtype=torch.float32)

    def __init__(self, file, degree, decay_rate):
        self.engine = Engine()
        self.engine.new(degree=degree, decay_rate=decay_rate)
        self.prep_data(file)
    
    def train(self, batch_size, num_epochs, lr):
        dataset = TensorDataset(self.X_ten, self.y_ten)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.engine.parameters(), lr=lr)

        for _ in tqdm(range(num_epochs)):
            self.engine.train()
            for X_batch, y_batch in loader:
                y_logits = self.engine(X_batch)
                loss = loss_fn(y_logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def get_engine(self):
        return self.engine

class Generator():
    def __init__(self, engine):
        self.engine = engine
    
    def generate(self):
        degree, decay_rate = self.engine.config.degree, self.engine.config.decay_rate

        self.engine.eval()
        with torch.inference_mode():
            newoun = ""

            X_vec = []
            for i in range(degree):
                for j in range(27):
                    X_vec.append(1 if j == 26 else 0)
            decay = decay_rate
            X_vec.append(1 - decay)

            for _ in range(20):
                X_ten = torch.tensor(X_vec).to(dtype=torch.float32).reshape(1, -1)
                y_logits = self.engine(X_ten)
                probs = torch.softmax(y_logits, dim=1).reshape(-1).tolist()

                value = random.random()
                accum, char_code = 0, 0
                while (char_code < 26) and ((accum + probs[char_code]) < value):
                    accum += probs[char_code]
                    char_code += 1

                if char_code == 26:
                    break
                else:
                    newoun += chr(ord("A") + char_code)

                for i in range(degree):
                    if i == degree - 1:
                        for j in range(27):
                            X_vec[(i * 27) + j] = 1 if j == char_code else 0
                    else:
                        for j in range(27):
                            X_vec[(i * 27) + j] = X_vec[((i + 1) * 27) + j]
                
                decay *= decay_rate
                X_vec[degree * 27] = 1 - decay
        
        return newoun.title()
