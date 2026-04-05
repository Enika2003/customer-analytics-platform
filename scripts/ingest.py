import pandas as pd
from sqlalchemy import create_engine, text

# --- Config ---
DB_URL = "postgresql+psycopg2://localhost/retail_analytics"
XLSX_PATH = "data/online_retail_II.xlsx"

def load_sheets(path):
    print("Reading sheet 1 (2009-2010)...")
    df1 = pd.read_excel(path, sheet_name="Year 2009-2010", dtype={"Customer ID": str})
    print(f"  {len(df1):,} rows")

    print("Reading sheet 2 (2010-2011)...")
    df2 = pd.read_excel(path, sheet_name="Year 2010-2011", dtype={"Customer ID": str})
    print(f"  {len(df2):,} rows")

    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined: {len(df):,} rows")
    return df

def clean(df):
    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    before = len(df)
    df = df.dropna(subset=["customer_id", "invoice"])
    print(f"Dropped {before - len(df):,} rows missing customer_id or invoice")

    # Remove cancellations (invoices starting with C)
    before = len(df)
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    print(f"Dropped {before - len(df):,} cancellation rows")

    # Remove rows with non-positive quantity or price
    before = len(df)
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    print(f"Dropped {before - len(df):,} rows with invalid quantity/price")

    # Add revenue column
    df["revenue"] = df["quantity"] * df["price"]

    print(f"Clean rows remaining: {len(df):,}")
    return df

def load_to_db(df, engine):
    print("Loading to PostgreSQL...")
    df.to_sql(
        "raw_transactions",
        engine,
        if_exists="replace",
        index=False,
        chunksize=10000,
        method="multi"
    )
    print("Done.")

def verify(engine):
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM raw_transactions")).scalar()
        sample = conn.execute(text("SELECT * FROM raw_transactions LIMIT 3")).fetchall()
    print(f"\nVerification: {count:,} rows in raw_transactions")
    print("Sample rows:")
    for row in sample:
        print(" ", row)

if __name__ == "__main__":
    engine = create_engine(DB_URL)
    df = load_sheets(XLSX_PATH)
    df = clean(df)
    load_to_db(df, engine)
    verify(engine)
