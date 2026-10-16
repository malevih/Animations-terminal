# Step 1 — messy user queries and the chunks that answer them (labelled before any rewriting or searching).
# Step 2 — rewrites written by Claude acting as the rewriting LLM, also before any searching.
# Variants: two paraphrases into formal/document language + one HyDE (a hypothetical answer sentence).
QR = [
 {"q": "cash when?", "rel": [12, 13], "rw": [
   "When does the buyer pay the supplier?",
   "payment terms and payment deadline",
   "The buyer pays the supplier within a number of days after receiving the invoice."]},
 {"q": "laptop dies after a month, what now", "rel": [16, 17, 18], "rw": [
   "What happens if a laptop breaks during the warranty period?",
   "warranty repair or replacement of a defective laptop",
   "If the laptop breaks within the warranty period, the supplier repairs or replaces it for free."]},
 {"q": "do i pay upfront", "rel": [14], "rw": [
   "Is an advance payment required?",
   "prepayment before delivery",
   "No advance payment is required; the buyer pays after delivery."]},
 {"q": "can a broke company apply", "rel": [6], "rw": [
   "Can a company in bankruptcy participate in the tender?",
   "bidder eligibility bankruptcy insolvency",
   "Companies that are bankrupt or insolvent cannot take part in the procurement."]},
 {"q": "who carries stuff inside", "rel": [10], "rw": [
   "Who is responsible for unloading and carrying the goods into the building?",
   "unloading boxes carrying into storage room",
   "The supplier unloads the boxes and carries them into the buyer's storage room."]},
 {"q": "how quick do they fix it", "rel": [18], "rw": [
   "How soon must warranty repairs start after a defect is reported?",
   "repair response time working days",
   "Repairs begin within a few working days after the buyer reports the defect."]},
 {"q": "tax?", "rel": [15], "rw": [
   "Does the contract price include VAT?",
   "value added tax included in the contract price",
   "The contract price already includes value added tax."]},
 {"q": "what specs", "rel": [1, 2], "rw": [
   "What are the technical requirements for the laptops?",
   "laptop processor memory RAM screen specifications",
   "Each laptop must have a multi-core processor, enough RAM and a large screen."]},
 {"q": "need proof of past work?", "rel": [4, 5], "rw": [
   "Does the bidder need to show experience with similar contracts?",
   "previous similar contract experience documents",
   "The bidder must attach copies of a similar contract completed in previous years."]},
 {"q": "big enough display?", "rel": [2], "rw": [
   "What is the minimum screen size?",
   "screen size in inches",
   "The screen must be at least fifteen inches."]},
]
