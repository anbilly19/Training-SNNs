import torch
import torch.nn as nn
from custom_neuron import CustomNeuron
from spikingjelly.activation_based import neuron, functional, surrogate, layer

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H

class ConvEnc(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.conv_fc = nn.Sequential(layer.Conv2d(1, args.n_channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(args.n_channels),
        CustomNeuron(tau=args.tau, surrogate_function=surrogate.ATan()))
        functional.set_step_mode(self, step_mode='m')
    
    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.n_channels, args.embed_dim,
                                kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim),
              requires_grad=True)  # Positional Embedding

    def forward(self, x):
        # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        x = self.conv1(x)
        # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.reshape([x.shape[0], self.args.embed_dim, -1])  
        x = x.transpose(1, 2)  # B E S -> B S E 
        # Adding classification token at the start of every sequence
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  
        x = x + self.pos_embedding  # Adding positional embedding
        return x


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads
        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)

    def forward(self, x):
        m, s, e = x.shape
        # B, Q, E -> B, Q, H, HE
        xq = self.queries(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  
        xq = xq.transpose(1, 2)  # B, Q, H, HE -> B, H, Q, HE
        # B, K, E -> B, K, H, HE
        xk = self.keys(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim) 
        xk = xk.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE
        # B, V, E -> B, V, H, HE
        xv = self.values(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  
        xv = xv.transpose(1, 2)  # B, V, H, HE -> B, H, V, HE

        xq = xq.reshape([-1, s, self.head_embed_dim])  # B, H, Q, HE -> (BH), Q, HE
        xk = xk.reshape([-1, s, self.head_embed_dim])  # B, H, K, HE -> (BH), K, HE
        xv = xv.reshape([-1, s, self.head_embed_dim])  # B, H, V, HE -> (BH), V, HE

        xk = xk.transpose(1, 2)  # (BH), K, HE -> (BH), HE, K
        x_attention = xq.bmm(xk)  # (BH), Q, HE  .  (BH), HE, K -> (BH), Q, K
        x_attention = torch.softmax(x_attention, dim=-1)

        x = x_attention.bmm(xv)  # (BH), Q, K . (BH), V, HE -> (BH), Q, HE
        # (BH), Q, HE -> B, H, Q, HE
        x = x.reshape([-1, self.n_attention_heads, s, self.head_embed_dim])  
        x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(m, s, e)  # B, Q, H, HE -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SelfAttention(args)
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)
        self.activation = CustomNeuron(tau=args.tau, surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        if self.eval:
            norm1 = self.norm1(x)
            att = self.attention(norm1)
            x = x + att # Skip connections
            x = self.dropout(x)
            norm2 = self.norm2(x)
            fc1 = self.fc1(norm2)
            spikes = self.activation(fc1)
            spikes_cpu = spikes.to('cpu')
            x = x + self.fc2(spikes)  #TODO Multiply
            return x
        elif self.train:
            x = x + self.attention(self.norm1(x)) # Skip connections
            x = self.dropout(x)
            x = x + self.fc2(self.activation(self.fc1(self.norm2(x)))) #TODO Multiply
            return x


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.activation = CustomNeuron(tau=args.tau, surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(args.embed_dim, args.n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbedLayer(args)
        self.encoder = nn.Sequential(*[Encoder(args) 
                                       for _ in range(args.n_layers)], nn.LayerNorm(args.embed_dim))
        self.norm = nn.LayerNorm(args.embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(args)

    def forward(self, x):
        x = self.embedding(x)
        if self.eval:
            x = self.encoder(x)
            x = self.norm(x)
            x = self.classifier(x)
            return x
        elif self.train:
            x = self.encoder(x)
            x = self.norm(x)
            x = self.classifier(x)
            return x
        
