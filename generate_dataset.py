"""
Generate a large, balanced, correctly-labeled customer review sentiment dataset.

Design goals (per project requirements):
- Sufficiently large (~3400 reviews) so the model can learn real patterns incl. negation.
- Balanced: exactly 50% positive / 50% negative.
- Correctly labeled: every review's label matches its expressed sentiment.
- No duplicate rows: every review string is unique (tracked via a `seen` set).
- Rich in negation examples (e.g. "not good", "does not work") so the model learns
  that "not good" is NEGATIVE, not positive.

The dataset is built from diverse templates with randomized slots so no two rows are
identical. Labels are assigned deterministically from the template's sentiment,
so there is no label noise.
"""
import random
import csv
import os

random.seed(42)

PRODUCTS = [
    "the product", "this item", "this device", "this gadget", "the phone",
    "the laptop", "the headphones", "the blender", "the shoes", "the watch",
    "the camera", "the TV", "the tablet", "the speaker", "the keyboard",
    "the mouse", "the bag", "the jacket", "the toy", "the kitchen appliance",
]

# ---------------- Positive building blocks ----------------
POS_ADJ = [
    "amazing", "excellent", "fantastic", "wonderful", "superb", "outstanding",
    "great", "perfect", "brilliant", "exceptional", "top-notch", "high quality",
    "durable", "reliable", "impressive", "stunning", "remarkable",
    "happy", "pleased", "delighted", "satisfied", "thrilled", "glad",
]
POS_VERB = ["love", "adore", "enjoy", "really like", "am impressed by", "am thrilled with"]
POS_SERVICE = [
    "The customer service was outstanding",
    "The support team was helpful",
    "Delivery was fast and reliable",
    "Shipping was quick",
    "Customer support responded quickly",
]
POS_EXTRA = [
    "Best purchase I've made this year.", "Highly recommended!", "Will buy again!",
    "Five stars!", "Worth every penny.", "Great value for money.",
    "Exactly what I was looking for.", "Exceeded my expectations.",
    "I'm very pleased with it.", "I would buy it again without hesitation.",
]

# ---------------- Negative building blocks ----------------
NEG_ADJ = [
    "terrible", "awful", "poor", "bad", "disappointing", "faulty", "defective",
    "broken", "cheap", "flimsy", "horrible", "rubbish", "worst", "useless",
    "unreliable", "subpar", "inferior",
]
NEG_VERB = ["hate", "dislike", "regret buying", "am disappointed by", "am not happy with", "cannot stand"]
NEG_ISSUE = [
    "broke after a few days", "stopped working", "arrived damaged",
    "does not work as advertised", "is defective", "has poor build quality",
    "is missing parts", "arrived late", "malfunctioned immediately",
]
NEG_EXTRA = [
    "Complete waste of money.", "Would not recommend to anyone.", "Do not buy this.",
    "Total garbage.", "Not worth the price.", "Very poor experience.",
    "Worst purchase of my life.", "Avoid this product at all costs.",
]

TARGET_PER_CLASS = 1700  # -> 3400 total, exactly balanced


def _pick(r, seq):
    return r.choice(seq)


# ---------------- Positive templates ----------------
def pos_t1(r):
    p = _pick(r, PRODUCTS); a = _pick(r, POS_ADJ); v = _pick(r, POS_VERB); e = _pick(r, POS_EXTRA)
    return f"{p.capitalize()} is {a}! I {v} it so much. {e}"

def pos_t2(r):
    a = _pick(r, POS_ADJ); s = _pick(r, POS_SERVICE); e = _pick(r, POS_EXTRA)
    return f"Absolutely {a} experience. {s} and the product exceeded my expectations. {e}"

def pos_t3(r):
    e = _pick(r, POS_EXTRA)
    return "Great value for money. Works exactly as described. " + e

def pos_t4(r):
    q = _pick(r, POS_ADJ); d = _pick(r, POS_SERVICE); e = _pick(r, POS_EXTRA)
    return f"I am really impressed with the {q} and {d.lower()}. Will buy again! {e}"

def pos_t5(r):
    e = _pick(r, POS_EXTRA)
    return "Excellent product! Easy to use and very durable. Five stars! " + e

def pos_t6(r):
    v = _pick(r, POS_VERB); e = _pick(r, POS_EXTRA)
    return f"Love this product! It has made my daily routine so much easier. I {v} it. {e}"

def pos_t7(r):
    a = _pick(r, POS_ADJ); e = _pick(r, POS_EXTRA)
    return f"Superb quality and {a} features. Worth every penny spent. {e}"

def pos_t8(r):
    e = _pick(r, POS_EXTRA)
    return "I'm extremely happy with this purchase. The product is exactly what I was looking for. " + e

def pos_t9(r):
    a = _pick(r, POS_ADJ); e = _pick(r, POS_EXTRA)
    return f"Outstanding performance and {a} design. I'm very impressed! {e}"

def pos_t10(r):
    e = _pick(r, POS_EXTRA)
    return "Best in class product. The quality is top-notch and delivery was quick. " + e

def pos_t11(r):
    q = _pick(r, POS_ADJ); s = _pick(r, POS_SERVICE)
    return f"Highly recommend! {q} quality, fast shipping, and {s.lower()}."

def pos_t12(r):
    a = _pick(r, POS_ADJ); e = _pick(r, POS_EXTRA)
    return f"Amazing value! Works perfectly and looks {a}. Will purchase more items from this brand. {e}"

def pos_t13(r):
    e = _pick(r, POS_EXTRA)
    return "This is the best product I have ever owned. Simply outstanding! " + e

def pos_t14(r):
    e = _pick(r, POS_EXTRA)
    return "Excellent build quality and very user-friendly. Loved it! " + e

def pos_t15(r):
    e = _pick(r, POS_EXTRA)
    return "I'm so glad I bought this. It's everything I needed and more. " + e

def pos_t16(r):
    e = _pick(r, POS_EXTRA)
    return "Five stars! Great product, fast delivery, and amazing customer service. " + e

def pos_t17(r):
    e = _pick(r, POS_EXTRA)
    return "I love everything about this product. Highly recommended to everyone! " + e

def pos_t18(r):
    e = _pick(r, POS_EXTRA)
    return "Fantastic purchase! Great quality and works perfectly. " + e

def pos_t19(r):
    e = _pick(r, POS_EXTRA)
    return "I'm very pleased with this product. Excellent quality and value! " + e

def pos_t20(r):
    a = _pick(r, POS_ADJ); e = _pick(r, POS_EXTRA)
    return f"The product is decent and {a}. It works well for basic use. {e}"

def pos_t21(r):
    e = _pick(r, POS_EXTRA)
    return "Good product overall. Does what it claims to do. Satisfied customer. " + e

def pos_t22(r):
    a = _pick(r, POS_ADJ); e = _pick(r, POS_EXTRA)
    return f"Best purchase ever! {a.capitalize()} quality, fast shipping, and excellent support. {e}"

def pos_t23(r):
    e = _pick(r, POS_EXTRA)
    return "I cannot say enough good things about this product. Absolutely wonderful! " + e

def pos_t24(r):
    a = _pick(r, POS_ADJ); e = _pick(r, POS_EXTRA)
    return f"I am very satisfied with this {a} purchase. Would definitely buy again. {e}"

def pos_t25(r):
    e = _pick(r, POS_EXTRA)
    return "The product works flawlessly and looks great. I'm thrilled with it! " + e

def pos_t26(r):
    e = _pick(r, POS_EXTRA)
    return "I am very happy with my purchase. " + e

def pos_t27(r):
    e = _pick(r, POS_EXTRA)
    return "I am so pleased with this product. Highly recommend! " + e

def pos_t28(r):
    e = _pick(r, POS_EXTRA)
    return "This product made me delighted. I am thrilled with it! " + e

def pos_t29(r):
    e = _pick(r, POS_EXTRA)
    return "I am satisfied with my purchase. Would definitely buy again. " + e

POS_TEMPLATES = [pos_t1, pos_t2, pos_t3, pos_t4, pos_t5, pos_t6, pos_t7, pos_t8,
                 pos_t9, pos_t10, pos_t11, pos_t12, pos_t13, pos_t14, pos_t15,
                 pos_t16, pos_t17, pos_t18, pos_t19, pos_t20, pos_t21, pos_t22,
                 pos_t23, pos_t24, pos_t25, pos_t26, pos_t27, pos_t28, pos_t29]




# ---------------- Negative templates ----------------
def neg_t1(r):
    p = _pick(r, PRODUCTS); a = _pick(r, NEG_ADJ); e = _pick(r, NEG_EXTRA)
    return f"{p.capitalize()} is {a}. Complete waste of money. {e}"

def neg_t2(r):
    p = _pick(r, PRODUCTS); a = _pick(r, NEG_ADJ); e = _pick(r, NEG_EXTRA)
    return f"Worst {p} ever bought. It {_pick(r, NEG_ISSUE)}. Very disappointed. {e}"

def neg_t3(r):
    e = _pick(r, NEG_EXTRA)
    return "Poor packaging and the item arrived damaged. Not happy with this purchase at all. " + e

def neg_t4(r):
    e = _pick(r, NEG_EXTRA)
    return "Complete waste of time and money. The product does not work as advertised. " + e

def neg_t5(r):
    e = _pick(r, NEG_EXTRA)
    return "Very bad experience. The seller never replied to my messages and the product is useless. " + e

def neg_t6(r):
    e = _pick(r, NEG_EXTRA)
    return "The worst customer service I have ever encountered. Rude and unhelpful staff. " + e

def neg_t7(r):
    e = _pick(r, NEG_EXTRA)
    return "Stopped working after just one week. Very poor build quality. " + e

def neg_t8(r):
    e = _pick(r, NEG_EXTRA)
    return "Do not buy this! It's a scam. The product is nothing like the pictures shown. " + e

def neg_t9(r):
    e = _pick(r, NEG_EXTRA)
    return "Horrible experience from start to finish. Late delivery and broken item. " + e

def neg_t10(r):
    e = _pick(r, NEG_EXTRA)
    return "Completely dissatisfied. The product is defective and support is non-existent. " + e

def neg_t11(r):
    e = _pick(r, NEG_EXTRA)
    return "I would give this zero stars if I could. Total garbage product. " + e

def neg_t12(r):
    e = _pick(r, NEG_EXTRA)
    return "Not worth the price. Cheap materials and poor performance. " + e

def neg_t13(r):
    e = _pick(r, NEG_EXTRA)
    return "The product arrived late and was missing parts. Very frustrating experience. " + e

def neg_t14(r):
    e = _pick(r, NEG_EXTRA)
    return "I regret buying this. It's poorly made and doesn't work properly. " + e

def neg_t15(r):
    e = _pick(r, NEG_EXTRA)
    return "Disappointing quality. The product feels cheap and flimsy. " + e

def neg_t16(r):
    e = _pick(r, NEG_EXTRA)
    return "Avoid this product at all costs. Worst purchase of my life. " + e

def neg_t17(r):
    e = _pick(r, NEG_EXTRA)
    return "The item stopped charging after three days. Complete junk. " + e

def neg_t18(r):
    e = _pick(r, NEG_EXTRA)
    return "Absolute rubbish. Do not waste your money on this terrible product. " + e

def neg_t19(r):
    e = _pick(r, NEG_EXTRA)
    return "Horrible quality control. Received a defective unit. Very upset. " + e

def neg_t20(r):
    e = _pick(r, NEG_EXTRA)
    return "Waste of money. The product broke within hours of using it. " + e

def neg_t21(r):
    e = _pick(r, NEG_EXTRA)
    return "The worst online shopping experience ever. Product never arrived and no refund. " + e

def neg_t22(r):
    e = _pick(r, NEG_EXTRA)
    return "Poorly designed and difficult to use. Not recommended. " + e

def neg_t23(r):
    e = _pick(r, NEG_EXTRA)
    return "Terrible product. Do not buy unless you want to waste your money. " + e

def neg_t24(r):
    e = _pick(r, NEG_EXTRA)
    return "The product is faulty and the company refuses to help. Awful service. " + e

def neg_t25(r):
    e = _pick(r, NEG_EXTRA)
    return "Avoid! Complete scam. Fake product and terrible customer service. " + e

def neg_t26(r):
    e = _pick(r, NEG_EXTRA)
    return "Extremely disappointing. The product is nothing like advertised. " + e

# Negation-heavy negative templates (the model must learn "not ..." = negative)
def neg_t27(r):
    p = _pick(r, PRODUCTS); e = _pick(r, NEG_EXTRA)
    return f"{p.capitalize()} is not good at all. {e}"

def neg_t28(r):
    p = _pick(r, PRODUCTS); e = _pick(r, NEG_EXTRA)
    return f"I do not like this {p}. {e}"

def neg_t29(r):
    p = _pick(r, PRODUCTS); e = _pick(r, NEG_EXTRA)
    return f"The {p} does not work properly. {e}"

def neg_t30(r):
    e = _pick(r, NEG_EXTRA)
    return "This product is not worth the money. " + e

def neg_t31(r):
    e = _pick(r, NEG_EXTRA)
    return "I am not happy with this purchase. " + e

def neg_t32(r):
    e = _pick(r, NEG_EXTRA)
    return "I would not recommend this product to anyone. " + e

def neg_t33(r):
    e = _pick(r, NEG_EXTRA)
    return "This is not what I expected. The product is disappointing. " + e

def neg_t34(r):
    p = _pick(r, PRODUCTS); e = _pick(r, NEG_EXTRA)
    return f"The {p} is not as described. Very unhappy. " + e

def neg_t35(r):
    e = _pick(r, NEG_EXTRA)
    return "I cannot get this product to work. It does not function at all. " + e

def neg_t36(r):
    e = _pick(r, NEG_EXTRA)
    return "Not impressed at all. The product is bad and I regret buying it. " + e

def neg_t37(r):
    v = _pick(r, NEG_VERB); p = _pick(r, PRODUCTS); e = _pick(r, NEG_EXTRA)
    return f"I {v} this {p}. {e}"

def neg_t38(r):
    v = _pick(r, NEG_VERB); e = _pick(r, NEG_EXTRA)
    return f"I {v} this product so much. {e}"

def neg_t39(r):
    v = _pick(r, NEG_VERB); e = _pick(r, NEG_EXTRA)
    return f"I really {v} this {_pick(r, PRODUCTS)}. {e}"

NEG_TEMPLATES = [neg_t1, neg_t2, neg_t3, neg_t4, neg_t5, neg_t6, neg_t7, neg_t8,
                 neg_t9, neg_t10, neg_t11, neg_t12, neg_t13, neg_t14, neg_t15,
                 neg_t16, neg_t17, neg_t18, neg_t19, neg_t20, neg_t21, neg_t22,
                 neg_t23, neg_t24, neg_t25, neg_t26, neg_t27, neg_t28, neg_t29,
                 neg_t30, neg_t31, neg_t32, neg_t33, neg_t34, neg_t35, neg_t36,
                 neg_t37, neg_t38, neg_t39]




def generate_class(r, templates, target, seen, label):
    """Generate `target` unique reviews of `label` using the given templates."""
    rows = []
    attempts = 0
    max_attempts = target * 40
    while len(rows) < target and attempts < max_attempts:
        attempts += 1
        text = _pick(r, templates)(r)
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, label))
    return rows


def main():
    r = random
    seen = set()
    pos_rows = generate_class(r, POS_TEMPLATES, TARGET_PER_CLASS, seen, "positive")
    neg_rows = generate_class(r, NEG_TEMPLATES, TARGET_PER_CLASS, seen, "negative")

    all_rows = pos_rows + neg_rows
    r.shuffle(all_rows)

    out_path = os.path.join("dataset", "reviews.csv")
    os.makedirs("dataset", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "sentiment"])
        writer.writerows(all_rows)

    n_pos = sum(1 for _, l in all_rows if l == "positive")
    n_neg = sum(1 for _, l in all_rows if l == "negative")
    print(f"Wrote {len(all_rows)} reviews to {out_path}")
    print(f"  positive: {n_pos}, negative: {n_neg}")
    print(f"  unique:   {len(seen)}")


if __name__ == "__main__":
    main()

