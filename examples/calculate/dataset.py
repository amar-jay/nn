import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

ALL_OPS = [
	"add", "sub", "mul", "div",
	"ln", "exp",
	"sin", "cos", "tan",
	"sinh", "cosh", "tanh",
	"arcsin", "arccos", "arctan",
	"arcsinh", "arccosh", "arctanh",
	"radtodeg", "degtorad",
]

class CalculationDataset(Dataset):
    def __init__(self, csv_path, filtered_ops: List[str] = ALL_OPS):
        self.df = pd.read_csv(csv_path, low_memory=False)
        # replace op with corresponding one-hot encoding
        self.op_map = {op: i for i, op in enumerate(ALL_OPS)}

        self.df["a"] = self.df["a"].to_numpy(dtype=float)
        # at time b might be "" or NaN for unary ops, replace with 0.0
        self.df["b"] = self.df["b"].replace("", 0.0).fillna(0.0)
        self.df["b"] = self.df["b"].to_numpy(dtype=float)
        self.df["result"] = self.df["result"].to_numpy(dtype=float)
        self.df['op'] = self.df['op'].map(self.op_map)
        self.df["valid"] = self.df["valid"].to_numpy(dtype=bool)

        # // filter ops that aren't add, sub, mul, div
        self.df = self.df[self.df['op'].isin([self.op_map[op] for op in filtered_ops])].reset_index(drop=True)
        # if there is any NaN within the dataframe raise an error
        if self.df.isnull().values.any():
            raise ValueError("Dataset contains NaN values after preprocessing.")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = torch.tensor([row["a"], row["b"], row["op"]], dtype=torch.float32)
        outputs = torch.tensor([row["result"], row["valid"]], dtype=torch.float32)

        return inputs, outputs

def collate_arithmetic(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.stack(ys, dim=0)
    return x_batch, y_batch

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    df = pd.read_csv("data/data.csv")
    # print(df.head())

    dataset = CalculationDataset("data/data.csv")
    print(f"Dataset size: {len(dataset)}")
    rand_idx = torch.randint(0, len(dataset), (1,)).item()
    sample_input, sample_output = dataset[rand_idx]
    print(f"Sample input: {sample_input}")
    print(f"Sample output: {sample_output}")