import unittest
from sympy import Q
import torch
import torch_optimizer

from src.config import PerceiverModelNet40Cfg, get_perceiver_model

class ModelTest(unittest.TestCase):
    
    def test_parameters_update(self):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        cfg = PerceiverModelNet40Cfg()
        model, _ = get_perceiver_model(cfg, device)
        input_tensor = torch.randint(0, 1220, (16, 2048, 3)).to(device)
        optimizer = torch_optimizer.Lamb(model.parameters(), lr=1e-3)
        output_tensor = model(input_tensor)
        loss = output_tensor[0].sum()
        loss.backward()
        optimizer.step()
        
        for param, name in zip(model.parameters(), model.state_dict()):
            if param.grad is None:
                print(name)

    def test_print_model(self):
        device = torch.device("cpu")
        cfg = PerceiverModelNet40Cfg()
        model, _ = get_perceiver_model(cfg, device)
        print(model)

if __name__ == "__main__":
    unittest.main()