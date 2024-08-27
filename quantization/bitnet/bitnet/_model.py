from dataclasses import dataclass
import torch.nn.functional as F
import torch
from torch import nn
from bitlinear import BitLinear


#v = m.encoder_decoder.is_torch_available()

#x = m.bert.is_torch_available()
#x = m.bert.BertConfig()
#x = m.bert.BertForTokenClassification()
#x = m.encoder_decoder.EncoderDecoderModel()

def inject_bitlinear(
    model: nn.Module,
    module_classes: list[str] | None = None,
):
    """
    `inject_bitlinear` is a function that replaces linear layers with BitLinear layers
    """

    _get_bitlinear = lambda linear: BitLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias = linear.bias is not None,
    )

    if isinstance(model, nn.Linear):
        return _get_bitlinear(model)

    if isinstance(model, (nn.Module, nn.ModuleDict)):
        for key, value in model._modules.items():
            if isinstance(value, nn.Linear) and (module_classes is None or key in module_classes):
                model._modules[key] = _get_bitlinear(value)
            elif isinstance(value, nn.Module):
                    inject_bitlinear(value)

    if isinstance(model, (nn.ModuleList, nn.Sequential)) :
        for sub_model in model:
            if isinstance(sub_model, nn.Linear) and (module_classes is None or sub_model in module_classes):
                sub_model = _get_bitlinear(sub_model)
            else:
                inject_bitlinear(sub_model)

    # resetting gradients
    for _, param in model.named_parameters():
        param.requires_grad = True
    return model


class RMSNorm(nn.Module):
    """
    RMSNorm is a layer normalization that is known for 
    rescaling but not recentering the input.
    This is from llama

    Paper: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, normalized_shape, eps=1e-05, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(*(normalized_shape,), device=device, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, input):
        output = self._norm(input.float()).type_as(input)
        return output * self.weight

@dataclass
class BertConfig:
    seq_length: int = 30 
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False # False: a bit better and faster
    linear_batch_groups: int = 1 # group batch size for BitLinear layers, 1 is the default
    weight_decay:float=0.01
    num_labels:int = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerBlock(nn.Module):
    def __init__(self, config: BertConfig, attention_mask=None) -> None:
        super().__init__()
        self.config = config
        self.attention_mask= attention_mask

        self.mlp = nn.ModuleDict(dict(
            _in = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            act= nn.GELU(),
            _out = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            dropout = nn.Dropout(config.dropout),
        )) 
        # feedforward layer

        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)
        # TODO: how to add attention mask to this??
        self.attention = nn.MultiheadAttention(config.n_embd, config.n_head, config.dropout, config.bias, device=config.device)

    def forward(self, x):
        mlp = lambda x: self.mlp.dropout(self.mlp._out(self.mlp.act(self.mlp._in(x)))) 
        # in the paper, normalization is after residual network,, however this seems like the modern approach in recent papers
        x = self.ln1(x)
        x = x + self.attention(x, x, x)[0]
        x = self.ln2(x)
        x = x + mlp(x)
        return x

class BERT(nn.Module):
    """
    Conventional BERT model
    """
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config

        self.emb = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(config.vocab_size, config.n_embd),
            pos_emb = nn.Embedding(config.seq_length, config.n_embd),
            dropout = nn.Dropout(config.dropout),
        ))

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])


    def forward(self, x, targets=None):
        assert x.size(1) <= self.config.seq_length, f"size of input ({x.shape=}) is than  sequence length ({self.config.seq_length})"

        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device) # shape (T,)
        emb = lambda x: self.emb.dropout(self.emb.tok_emb(x) + self.emb.pos_emb(pos))
        x = emb(x)

        for layer in self.layers:
            x = layer(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            #logits = self.class_linear(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return x, loss

class BertForTokenClassification(nn.Module):
    """
    This is a BERT model for NER
    """
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config

        self.model = BERT(config)

        self.classifier = nn.ModuleDict(dict(
            #ln = nn.LayerNorm(config.vocab_size) # not neccessart for now
            tanh = nn.Tanh(),
            dropout = nn.Dropout(config.dropout/5), # I think dropout after tanh because of the small dimensionality
            class_linear = nn.Linear(config.n_embd, config.num_labels)
        ))

    def forward(self, x, targets=None):
        x, loss = self.model(x, None)
        logits = self.classifier.class_linear(self.classifier.dropout(x))
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            #logits = self.class_linear(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

if __name__ == '__main__':
    import argparse

    # this is a model to run a NERT model using BERT
    # with a vocab_size of 10k
    # and a sequence length (max_length) of 512
    # and a batch size of 4
    parser = argparse.ArgumentParser(description='A demo BERT NER')
    config = BertConfig()

    parser.add_argument('--vocab_size', type=int, default=config.vocab_size, help='vocab size')
    parser.add_argument('--seq_length', type=int, default=config.seq_length, help='max token length')
    parser.add_argument('--dropout', type=float, default=config.dropout, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay, help='weight decay')
    parser.add_argument('--learning_rate', type=float, default=1, help='learning rate')
    parser.add_argument('--num_labels', type=int, default=5, help='the number of labels')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    args = parser.parse_args()


    config = BertConfig(
        vocab_size=args.vocab_size, 
        seq_length=args.seq_length,
        dropout=args.dropout,
        #num_labels=args.num_labels
    )
    weight_decay=args.weight_decay
    lr=args.learning_rate
    epoch = args.epochs

    model = BertForTokenClassification(config)
    data = torch.rand((2, args.seq_length)).to(torch.int64)
    print(">>> ", model(data)[0].shape)

    model = inject_bitlinear(model)
    print(">>> ", model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Epoch {args.epochs}")
    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")
    print(f"Config: {config}")

    for i in range(1):
        if i % 10 == 0:
            print(f"Saving model at epoch {i}") 



    print(args)
