import csv

# Check "very" in reviews
very_pos = 0
very_neg = 0
good_pos = 0
good_neg = 0
very_good_together = 0

with open("dataset/reviews.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["review"].lower()
        label = row["sentiment"]
        has_very = "very" in text
        has_good = "good" in text
        
        if has_very:
            if label == "positive":
                very_pos += 1
            else:
                very_neg += 1
        if has_good:
            if label == "positive":
                good_pos += 1
            else:
                good_neg += 1
        if has_very and has_good:
            very_good_together += 1

print(f"Reviews with 'very': positive={very_pos}, negative={very_neg}")
print(f"Reviews with 'good': positive={good_pos}, negative={good_neg}")
print(f"Reviews with both 'very' and 'good': {very_good_together}")

# Show some examples of "very" in positive reviews
print("\nExamples of 'very' in positive reviews:")
count = 0
with open("dataset/reviews.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "very" in row["review"].lower() and row["sentiment"] == "positive":
            print(f"  '{row['review'][:100]}...'")
            count += 1
            if count >= 5:
                break

# Show examples of "good" in positive reviews (without "not")
print("\nExamples of 'good' in positive reviews (without 'not'):")
count = 0
with open("dataset/reviews.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["review"].lower()
        if "good" in text and row["sentiment"] == "positive" and "not good" not in text:
            print(f"  '{row['review'][:100]}...'")
            count += 1
            if count >= 5:
                break