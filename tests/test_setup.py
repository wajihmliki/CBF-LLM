# test_setup.py
# Run this to verify your setup works.
# You DON'T need a GPU for this — it just checks imports.

from lcf import BaseLanguageCF, SentimentLCF, ToxicityLCF, TopicLCF

print("All imports OK")


# If you have GPU and want to test the actual L-CF:
lcf = SentimentLCF(device="cuda")  # or "cpu"
score = lcf.get("I love this movie!")
print(f"h('I love this movie!') = {score}")
score2 = lcf.get("This is terrible and I hate it.")
print(f"h('This is terrible...') = {score2}")

# --- Test Sentiment L-CF ---
print("\n--- Sentiment L-CF ---")
sent_lcf = SentimentLCF(device="cuda")
print(f"h('I love this!') = {sent_lcf.get('I love this!'):.4f}")
print(f"h('I hate this.') = {sent_lcf.get('I hate this.'):.4f}")

# --- Test Toxicity L-CF ---
print("\n--- Toxicity L-CF ---")
tox_lcf = ToxicityLCF(device="cuda")
print(f"h('Have a great day!') = {tox_lcf.get('Have a great day!'):.4f}")
print(f"h('You are an idiot and I will hurt you') = {tox_lcf.get('You are an idiot and I will hurt you'):.4f}")

# --- Test Topic L-CF ---
print("\n--- Topic L-CF ---")
topic_lcf = TopicLCF(topic="This text is about science and research.", device="cuda")
print(f"h('Quantum physics explores subatomic particles') = {topic_lcf.get('Quantum physics explores subatomic particles'):.4f}")
print(f"h('The football match was exciting') = {topic_lcf.get('The football match was exciting'):.4f}")