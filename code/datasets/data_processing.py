# Import Necessary Libraries
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from sklearn.model_selection import train_test_split


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.drop("Unnamed: 0", inplace=True, axis=1)
    return df


def clean_text(sent):
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    words = [word for word in words if word not in stop_words]

    return " ".join(words).lower()


def clean_data(df):
    """Perform data cleaning: handle missing values and remove outliers."""
    # Example cleaning: Drop rows with missing target values
    df['text'] = df["text"].apply(clean_text)
    return df

def label_encoding(df, dis2idx):
    diseases = df["label"].unique()
    #
    # idx2dis = {k: v for k, v in enumerate(diseases)}
    # dis2idx = {v: k for k, v in idx2dis.items()}
    df["label"] = df["label"].apply(lambda x: dis2idx[x])
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split the data into train and test datasets."""
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=test_size, random_state=random_state)
    train_df = pd.DataFrame()
    train_df["text"] = X_train
    train_df["label"] = y_train
    test_df = pd.DataFrame()
    test_df["text"] = X_test
    test_df["label"] = y_test
    return train_df, test_df

def save_data(train_df, test_df, train_filepath, test_filepath):
    """Save the train and test datasets to CSV files."""
    train_df.to_csv(train_filepath, index=False)
    test_df.to_csv(test_filepath, index=False)

def main():
    # File paths
    raw_data_path = 'data/raw/Symptom2Disease.csv'
    train_data_path = 'data/processed/train.csv'
    test_data_path = 'data/processed/test.csv'

    # Load the raw data
    data = load_data(raw_data_path)

    # Clean the data
    cleaned_data = clean_data(data)
    diseases = cleaned_data["label"].unique()
    idx2dis = {k: v for k, v in enumerate(diseases)}
    dis2idx = {v: k for k, v in idx2dis.items()}
    encoded = label_encoding(cleaned_data, dis2idx)
    train_df, test_df = split_data(encoded)

    # Save the processed data
    save_data(train_df, test_df, train_data_path, test_data_path)

if __name__ == '__main__':
    main()
