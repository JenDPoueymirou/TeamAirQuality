import pandas as pd

def load_pm25():
    return pd.read_csv("data/pm25_data.csv")

def load_asthma():
    return pd.read.csv("data/asthma_er_visits.csv")

def clean_columns(df):
    df.columns = df.columns.str.lower().str.replace(" ". " ")
    return df

def merge_data():
    pm = clean_columns(load_pm25())
    asthma = clean_columns(load_asthma())

    merged = pm.merge(asthma, on = ["neighborhood", "year"], how="inner")

    merged.to_csv("data/merged_data.csv", index=False)
    print("Saved merged_data.csv")

if __name__ == "__main__":
    merge_data()
    