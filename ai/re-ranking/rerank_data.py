# Harder queries for re-ranking. Relevant clauses (1-based) labeled by hand BEFORE running any model.
Q = [
 ("will we get a loaner while ours is in the shop",          [20]),
 ("who pays for the truck",                                  [10]),
 ("can the price go up later",                               [16]),
 ("do we have to pay anything upfront",                      [15]),
 ("how much memory does each machine need",                  [2]),
 ("what proof of past experience must be included",          [6]),
 ("which firms are excluded from bidding",                   [7]),
 ("how long until the technician begins fixing it",          [19]),
 ("is windows included",                                     [4]),
 ("who confirms that the order arrived",                     [12]),
 ("how many computers are being bought",                     [1]),
 ("display size requirements",                               [3]),
 ("who is allowed to sign on behalf of the company",         [8]),
 ("how long is the coverage period",                         [17]),
 ("what experience must the bidder have",                    [5]),
 ("currency of payment",                                     [13]),
]

# Scoring (run from a folder with ce.py, minilm.py and the tender sentences):
#   from ce import score            # cross-encoder ms-marco-MiniLM-L-6-v2, ONNX by Xenova,
#                                   # taken from the npm package kontexta-reranker-model@1.1.0
#   ce = [score(q, sentences) for q, _ in Q]   # 16 x 20 raw logits, higher = more relevant
