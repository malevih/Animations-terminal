import json, numpy as np
from minilm import embed
DOC = [
 ("scope", "The buyer is purchasing twenty laptops for the finance department."),
 ("scope", "Each laptop must have at least an eight-core processor and sixteen gigabytes of RAM."),
 ("scope", "The screen must be at least fifteen inches with a matte finish."),
 ("scope", "A licensed operating system must be preinstalled on every device."),
 ("bidder", "The bidder must have completed at least one similar contract in the last three years."),
 ("bidder", "Copies of that contract and the signed acceptance certificates must be attached to the bid."),
 ("bidder", "Companies in bankruptcy or liquidation proceedings are not allowed to participate."),
 ("bidder", "The bid is signed by the company director or a person with a power of attorney."),
 ("delivery", "Delivery must be completed within thirty calendar days after the contract is signed."),
 ("delivery", "The supplier transports the goods to the buyer's warehouse at its own expense."),
 ("delivery", "Unloading the boxes and carrying them into the storage room is the supplier's job."),
 ("delivery", "Receipt of the goods is confirmed by a delivery note signed by the warehouse keeper."),
 ("payment", "Payment is made in hryvnias by bank transfer to the supplier's account."),
 ("payment", "The money is transferred within ten banking days after the invoice is received."),
 ("payment", "No advance payment is provided under this contract."),
 ("payment", "The contract price includes value added tax and cannot be increased."),
 ("warranty", "The warranty period for the laptops is at least two years."),
 ("warranty", "If a device breaks down, the supplier repairs or replaces it free of charge."),
 ("warranty", "Repairs must start within five working days after the buyer reports a defect."),
 ("warranty", "A replacement device is provided while the original is being repaired."),
]
X = embed([s for _, s in DOC])
true = [i for i in range(len(DOC) - 1) if DOC[i][0] != DOC[i + 1][0]]
for buf in (0, 1):
    B = np.array([X[max(0, i - buf):i + buf + 1].mean(0) for i in range(len(X))]); B /= np.linalg.norm(B, axis=1, keepdims=True)
    d = [1 - float(B[i] @ B[i + 1]) for i in range(len(X) - 1)]
    top = sorted(sorted(range(len(d)), key=lambda i: -d[i])[:4])
    thr = np.percentile(d, 80); cuts = [i for i, x in enumerate(d) if x > thr]
    print(f'buf{buf} true {true} top4 {top} | p80 cuts {cuts}')
    print('   ', ' '.join(f'{x:.2f}' for x in d))
json.dump({'sentences': [s for _, s in DOC], 'topics': [t for t, _ in DOC], 'emb': np.round(X, 4).tolist()}, open('chunk_emb.json', 'w'))
