"""Model micro."""
import random
from autograd import matrix
from tokens import Tokenizer


# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    """Return linear."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    """Return softmax."""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """Return rmsnorm."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


class Model:  # pylint: disable=too-many-instance-attributes
    """Micro gpt."""

    n_layer = 1  # depth of the transformer neural network (number of layers)
    n_embd = 16  # width of the network (embedding dimension)
    # note: the longest name is 15 characters
    block_size = 16  # maximum context length of the attention window
    n_head = 4  # number of attention heads
    head_dim = n_embd // n_head  # derived dimension of each head

    def __init__(self):
        """Initialize the parameters, to store the knowledge of the model."""
        self.state_dict = {
          'wpe': matrix(self.block_size, self.n_embd),
        }
        self.tok = None

        for i in range(self.n_layer):
            l_name = 'layer{}.'.format(i)
            self.state_dict[l_name + 'attn_wq'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[l_name + 'attn_wk'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[l_name + 'attn_wv'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[l_name + 'attn_wo'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[l_name + 'mlp_fc1'] = matrix(4 * self.n_embd, self.n_embd)
            self.state_dict[l_name + 'mlp_fc2'] = matrix(self.n_embd, 4 * self.n_embd)

    def gpt(self, token_id, pos_id, keys, values):  # pylint: disable=too-many-locals
        """Return gpt."""
        tok_emb = self.state_dict['wte'][token_id]  # token embedding
        pos_emb = self.state_dict['wpe'][pos_id]  # position embedding
        x = [t + p for t, p in zip(tok_emb, pos_emb)]  # joint token and position embedding
        x = rmsnorm(x)  # note: not redundant due to backward pass via the residual connection

        for li in range(self.n_layer):
            l_name = 'layer{}.'.format(li)

            # 1) Multi-head Attention block
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, self.state_dict[l_name + 'attn_wq'])
            k = linear(x, self.state_dict[l_name + 'attn_wk'])
            v = linear(x, self.state_dict[l_name + 'attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs + self.head_dim]
                k_h = [ki[hs:hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + self.head_dim] for vi in values[li]]
                attn_logits = [
                  sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                  for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                  sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                  for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, self.state_dict[l_name + 'attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]

            # 2) MLP block
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, self.state_dict[l_name + 'mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, self.state_dict[l_name + 'mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        return linear(x, self.state_dict['lm_head'])  # logits

    def learn(self, docs, progress_bar=None):  # pylint: disable=too-many-locals
        """Train model with given list of docs."""
        self.tok = Tokenizer(docs)
        self.state_dict.update({
          'wte': matrix(self.tok.size, self.n_embd),
          'lm_head': matrix(self.tok.size, self.n_embd),
        })
        # flatten params into a single list[Value]
        params = [p for mat in self.state_dict.values() for row in mat for p in row]

        # Let there be Adam, the blessed optimizer and its buffers
        learning_rate = 0.01
        beta1 = 0.85
        beta2 = 0.99
        eps_adam = 1e-8
        m = [0.0] * len(params)  # first moment buffer
        v = [0.0] * len(params)  # second moment buffer

        num_steps = len(docs)

        for step, doc in enumerate(docs):
            tokens = self.tok.tokenize(doc)
            n = min(self.block_size, len(tokens) - 1)

            # Forward the token sequence through the model,
            # building up the computation graph all the way to the loss
            keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
            losses = []
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = self.gpt(token_id, pos_id, keys, values)
                probs = softmax(logits)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)
            loss = (1 / n) * sum(losses)  # final average loss over the document sequence. May yours be low.

            # Backward the loss, calculating the gradients with respect to all model parameters
            loss.backward()

            # Adam optimizer update: update the model parameters based on the corresponding gradients
            lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
            for i, p in enumerate(params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
                p.grad = 0

            if progress_bar:
                if progress_bar(num_steps, step, "loss {:.4f}".format(loss.data)):
                    return len(params)

        return len(params)

    def ask(self, temperature=1):  # in (0, 1], control the "creativity" of generated text, low to high
        """Inference: may the model babble back to us."""
        keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
        token_id = self.tok.bos
        sample = []
        for pos_id in range(self.block_size):
            logits = self.gpt(token_id, pos_id, keys, values)
            probs = softmax([logit / temperature for logit in logits])
            token_id = random.choices(range(self.tok.size), weights=[p.data for p in probs])[0]
            if token_id == self.tok.bos:
                break
            sample.append(self.tok.uchars[token_id])

        return ''.join(sample)
