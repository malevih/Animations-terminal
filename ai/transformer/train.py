"""Train a tiny word-level decoder-only transformer and export weights as JSON."""
import json, itertools, random
import numpy as np
import jax, jax.numpy as jnp

A = ['cat', 'dog', 'fox']; F = ['fish', 'meat', 'bread']
T = ['bus', 'train', 'tram']; P = ['city', 'town', 'village']
S = []
for a in A:
    for f in F:
        S.append(f'the {a} eats {f} .')
        S.append(f'the {a} ate the {f} because it was hungry .')
    S.append(f'the {a} sleeps at home .')
    S.append(f'the {a} runs fast .')
for t in T:
    for p in P:
        S.append(f'the {t} goes to the {p} .')
    S.append(f'the {t} carries people .')
for p in P:
    S.append(f'people live in the {p} .')
    S.append(f'the {p} is big .')

BOS = '<s>'
words = sorted({w for s in S for w in s.split()})
VOCAB = [BOS] + words
ix = {w: i for i, w in enumerate(VOCAB)}
CTX = 12
seqs = [[ix[BOS]] + [ix[w] for w in s.split()] for s in S]
assert max(len(s) for s in seqs) <= CTX + 1, max(len(s) for s in seqs)

# training rows: input x[:-1], target x[1:], padded; mask pads
X = np.zeros((len(seqs), CTX), np.int32); Y = np.zeros((len(seqs), CTX), np.int32); M = np.zeros((len(seqs), CTX), np.float32)
for r, s in enumerate(seqs):
    inp, tgt = s[:-1], s[1:]
    X[r, :len(inp)] = inp; Y[r, :len(tgt)] = tgt; M[r, :len(tgt)] = 1

V, D, H, L, FF = len(VOCAB), 32, 2, 2, 128
key = jax.random.PRNGKey(0)
def init():
    ks = iter(jax.random.split(key, 64))
    n = lambda *shape, s=0.08: jax.random.normal(next(ks), shape) * s
    p = {'tok': n(V, D), 'pos': n(CTX, D), 'lnf_g': jnp.ones(D), 'lnf_b': jnp.zeros(D), 'out_w': n(D, V), 'out_b': jnp.zeros(V), 'layers': []}
    for _ in range(L):
        p['layers'].append({'ln1_g': jnp.ones(D), 'ln1_b': jnp.zeros(D), 'wq': n(D, D), 'wk': n(D, D), 'wv': n(D, D), 'wo': n(D, D),
                            'ln2_g': jnp.ones(D), 'ln2_b': jnp.zeros(D), 'w1': n(D, FF), 'b1': jnp.zeros(FF), 'w2': n(FF, D), 'b2': jnp.zeros(D)})
    return p

def ln(x, g, b):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * g + b

def forward(p, x):
    n = x.shape[-1]
    h = p['tok'][x] + p['pos'][:n]
    mask = jnp.tril(jnp.ones((n, n), bool))
    for ly in p['layers']:
        a = ln(h, ly['ln1_g'], ly['ln1_b'])
        q, k, v = a @ ly['wq'], a @ ly['wk'], a @ ly['wv']
        dh = D // H
        q = q.reshape(*q.shape[:-1], H, dh); k = k.reshape(*k.shape[:-1], H, dh); v = v.reshape(*v.shape[:-1], H, dh)
        att = jnp.einsum('...ihd,...jhd->...hij', q, k) / jnp.sqrt(dh)
        att = jnp.where(mask, att, -1e9)
        att = jax.nn.softmax(att, -1)
        o = jnp.einsum('...hij,...jhd->...ihd', att, v).reshape(*h.shape)
        h = h + o @ ly['wo']
        m = ln(h, ly['ln2_g'], ly['ln2_b'])
        h = h + jax.nn.gelu(m @ ly['w1'] + ly['b1'], approximate=True) @ ly['w2'] + ly['b2']
    h = ln(h, p['lnf_g'], p['lnf_b'])
    return h @ p['out_w'] + p['out_b']

def loss_fn(p, x, y, m):
    logits = forward(p, x)
    lp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(lp, y[..., None], -1)[..., 0]
    return (nll * m).sum() / m.sum()

@jax.jit
def step(p, opt, x, y, m, t):
    l, g = jax.value_and_grad(loss_fn)(p, x, y, m)
    b1, b2, lr = .9, .98, 3e-3
    mu = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, opt['m'], g)
    nu = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, opt['v'], g)
    p = jax.tree.map(lambda w, a, b: w - lr * (a / (1 - b1 ** t)) / (jnp.sqrt(b / (1 - b2 ** t)) + 1e-8), p, mu, nu)
    return p, {'m': mu, 'v': nu}, l

p = init()
opt = {'m': jax.tree.map(jnp.zeros_like, p), 'v': jax.tree.map(jnp.zeros_like, p)}
Xj, Yj, Mj = jnp.array(X), jnp.array(Y), jnp.array(M)
for t in range(1, 1501):
    p, opt, l = step(p, opt, Xj, Yj, Mj, t)
    if t in (1, 100, 500, 1000, 1500): print('step', t, 'loss', float(l))

n_params = sum(int(np.prod(a.shape)) for a in jax.tree.leaves(p))
print('params', n_params, 'vocab', V, 'sentences', len(S))

# sanity: next-token distributions
def probs(prompt):
    ids = [ix[BOS]] + [ix[w] for w in prompt.split()]
    lg = forward(p, jnp.array(ids))[-1]
    pr = np.array(jax.nn.softmax(lg))
    top = pr.argsort()[::-1][:4]
    return ', '.join(f'{VOCAB[i]}:{pr[i]:.2f}' for i in top)
for pr in ['the', 'the cat', 'the cat eats', 'the bus goes to the', 'the dog ate the meat because', 'people live in the']:
    print(f'{pr!r:32} -> {probs(pr)}')

r4 = lambda a: np.round(np.asarray(a, dtype=np.float64), 4).tolist()
out = {'vocab': VOCAB, 'ctx': CTX, 'd': D, 'heads': H, 'layers_n': L, 'ff': FF, 'params': n_params, 'sentences': len(S),
       'tok': r4(p['tok']), 'pos': r4(p['pos']), 'lnf_g': r4(p['lnf_g']), 'lnf_b': r4(p['lnf_b']), 'out_w': r4(p['out_w']), 'out_b': r4(p['out_b']),
       'layers': [{k: r4(v) for k, v in ly.items()} for ly in p['layers']]}
json.dump(out, open('tf_weights.json', 'w'), separators=(',', ':'))

# reference logits with the ROUNDED weights, for checking the JS forward pass
pr_ = jax.tree.map(lambda a: jnp.array(np.round(np.asarray(a, dtype=np.float64), 4), dtype=jnp.float32), p)
ids = [ix[BOS]] + [ix[w] for w in 'the dog ate the meat because'.split()]
ref = np.array(forward(pr_, jnp.array(ids))[-1])
json.dump({'prompt': 'the dog ate the meat because', 'logits': ref.round(5).tolist()}, open('tf_ref.json', 'w'))
print('saved', len(open('tf_weights.json').read()) // 1024, 'KB')
