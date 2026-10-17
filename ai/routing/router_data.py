# labelled before running anything: test questions with the clauses a full answer needs, plus router training examples
TEST = [
 ("How many laptops are being purchased?", [0]), ("What screen size is required?", [2]), ("Is an operating system included?", [3]),
 ("Who signs the bid?", [7]), ("Where are the goods delivered?", [9]), ("Is there an advance payment?", [14]), ("How long is the warranty?", [16]),
 ("Can a bankrupt company with relevant experience take part?", [4, 6]), ("Compare the payment deadline with the delivery deadline.", [8, 13]),
 ("If a laptop breaks, how fast is it fixed and what does the buyer use meanwhile?", [18, 19]),
 ("What documents must a bidder attach and who signs the bid?", [5, 7]), ("Summarize the money terms: currency, timing and VAT.", [12, 13, 15]),
 ("What has to happen between delivery and payment?", [11, 13]), ("Which delivery costs and work does the supplier cover?", [9, 10]),
]
TRAIN = [
 ("What processor is needed?", "easy"), ("What currency is used for payment?", "easy"), ("Who unloads the boxes?", "easy"), ("When must delivery be finished?", "easy"),
 ("Explain all obligations of the supplier after a defect is reported.", "hard"), ("Which bidder requirements and payment terms could make an offer fail?", "hard"),
 ("Compare the warranty terms with the payment terms.", "hard"), ("List every deadline in the contract and how they relate to each other.", "hard"),
]
