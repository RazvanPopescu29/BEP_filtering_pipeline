import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_lengths(csv_path):
    print("Loading data")
    df = pd.read_csv(csv_path, usecols=["text"], dtype=str)
    df = df.dropna(subset=["text"])
    
    print("Calculating lengths")
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["char_count"] = df["text"].apply(len)
    
    percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    
    print("      WORD COUNT DISTRIBUTION         ")

    print(df["word_count"].describe(percentiles=percentiles))
    
    print("    CHARACTER COUNT DISTRIBUTION      ")
    print(df["char_count"].describe(percentiles=percentiles))

    cap_limit = 100 
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df["word_count"] <= cap_limit]["word_count"], bins=50, kde=True, color="blue")
    plt.title(f"Distribution of Message Word Counts (Showing 0 to {cap_limit} words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Messages")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    analyze_lengths("unfiltered-messages.csv")