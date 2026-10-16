import json, numpy as np
import sys; sys.path.insert(0, "../chunking")
from minilm import embed  # see ai/chunking/minilm.py
QS = [  # query, relevant chunk indices (any of them answers the question)
 ('when will the supplier get paid', [12, 13]), ('what if the computer stops working', [17, 18, 19]), ('deadline of ten days', [13]),
 ('who can take part in the tender', [4, 5, 6, 7]), ('how fast is the shipping', [8]), ('hryvnia', [12]), ('power of attorney', [7]),
 ('VAT included in price', [15]), ('eight-core processor', [1]), ('matte screen', [2]), ('bankruptcy', [6]),
 ('sixteen gigabytes', [1]), ('is there an advance payment', [14]), ('who unloads the boxes', [10]),
]
QE = embed([q for q, _ in QS])
json.dump({'queries': [q for q, _ in QS], 'rel': [r for _, r in QS], 'qemb': np.round(QE, 4).tolist()}, open('hyb_q.json', 'w'))
print('ok', len(QS))
