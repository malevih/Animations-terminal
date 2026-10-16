# Written in this order: questions -> which chunks a correct answer needs -> rewrites -> answers. Search runs after.
RAG = [
 {"q": "How and when does the supplier get paid?", "need": [12, 13, 14],
  "rw": ["What are the payment terms of the contract?", "payment bank transfer invoice advance",
         "The buyer pays by bank transfer within several days after the invoice; no advance payment."],
  "answer": [
    {"text": "The supplier is paid in hryvnias by bank transfer", "cite": 12, "quote": "by bank transfer"},
    {"text": "within ten banking days after the invoice is received", "cite": 13, "quote": "within ten banking days after the invoice is received"},
    {"text": "There is no advance payment.", "cite": 14, "quote": "No advance payment"}],
  "norag": [
    {"text": "Payment is usually made within 30 days after delivery", "quote": "30 days after delivery"},
    {"text": "often with a 30% advance payment.", "quote": "30% advance"}]},
 {"q": "What happens if one of the laptops breaks?", "need": [16, 17, 18, 19],
  "rw": ["What are the warranty obligations if a device breaks?", "warranty repair replacement defect",
         "During the warranty the supplier repairs or replaces the broken laptop free of charge."],
  "answer": [
    {"text": "The warranty lasts at least two years.", "cite": 16, "quote": "at least two years"},
    {"text": "The supplier repairs or replaces the device free of charge", "cite": 17, "quote": "repairs or replaces it free of charge"},
    {"text": "and must start repairs within five working days of the report.", "cite": 18, "quote": "within five working days"},
    {"text": "A replacement device is provided while it is being repaired.", "cite": 19, "quote": "replacement device is provided"}],
  "norag": [
    {"text": "The standard warranty is 12 months.", "quote": "12 months"},
    {"text": "The buyer sends the laptop to an authorized service center.", "quote": "service center"}]},
 {"q": "Can the delivery deadline be extended?", "need": [8],
  "rw": ["Is it possible to extend the delivery period?", "delivery deadline extension",
         "The delivery period may be extended by agreement of the parties."],
  "answer": [
    {"text": "Delivery must be completed within thirty calendar days after the contract is signed.", "cite": 8, "quote": "within thirty calendar days"},
    {"text": "The document does not say that this deadline can be extended.", "absent": ["extend", "extension", "prolong"]}],
  "norag": [
    {"text": "Yes, the deadline can be extended by mutual agreement of the parties", "quote": "extended by mutual agreement"},
    {"text": "if the supplier sends a written request in advance.", "quote": "written request"}]},
]
