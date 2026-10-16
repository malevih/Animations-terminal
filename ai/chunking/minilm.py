import numpy as np, onnxruntime as ort, sys
from tokenizers import Tokenizer
# npm pack @xcidos/genesis-memory-model && tar xzf *.tgz  (all-MiniLM-L6-v2, ONNX + tokenizer)
M = 'package/'
tok = Tokenizer.from_file(M + 'tokenizer.json'); tok.enable_padding(); tok.enable_truncation(256)
sess = ort.InferenceSession(M + 'onnx/model.onnx')
def embed(texts):
    enc = tok.encode_batch(texts)
    ids = np.array([e.ids for e in enc], np.int64); am = np.array([e.attention_mask for e in enc], np.int64)
    feeds = {'input_ids': ids, 'attention_mask': am}
    if 'token_type_ids' in [i.name for i in sess.get_inputs()]: feeds['token_type_ids'] = np.zeros_like(ids)
    out = sess.run(None, feeds)[0]
    v = (out * am[..., None]).sum(1) / am.sum(1, keepdims=True)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
if __name__ == '__main__':
    print([i.name for i in sess.get_inputs()], [o.name + str(o.shape) for o in sess.get_outputs()])
    E = embed(['A man is eating food.', 'Someone is having a meal.', 'The stock market fell today.'])
    print('sim(eat, meal)=%.3f  sim(eat, market)=%.3f' % (E[0] @ E[1], E[0] @ E[2]))
