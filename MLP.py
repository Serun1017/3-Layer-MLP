import torch.nn as nn

class MLP(nn.Module) :
    # input_Size, hidden_size, output_size initialize 
    def __init__(self, input_size=784, hidden_size=256, output_size=10) :
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # input Layer
            nn.Linear(input_size, hidden_size),

            # Activation Function
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(), 

            # hidden Layer
            nn.Linear(hidden_size, hidden_size),

            # Activation Function
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            
            # output Layer
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x) :
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits