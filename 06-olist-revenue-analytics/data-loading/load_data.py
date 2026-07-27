"""
Load Olist CSVs into local PostgreSQL (olist_analytics database).

Run from project root with venv activated:
    python data-loading/load_data.py

Requires:
    - PostgreSQL running locally with database 'olist_analytics' created
    - psycopg2-binary, pandas installed
    - CSVs present in data-loading/raw/

You'll be prompted for your postgres password when this runs.
"""
import pandas as pd
import psycopg2
from psycopg2 import sql
import os
import getpass

RAW_DIR = os.path.join("data-loading", "raw")
SCHEMA_FILE = os.path.join("data-loading", "create_tables.sql")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "olist_analytics",
    "user": "postgres",
}


def get_connection():
    password = getpass.getpass("Postgres password for user 'postgres': ")
    return psycopg2.connect(password=password, **DB_CONFIG)


def run_schema(conn):
    print("Creating schema (tables, PKs, FKs)...")
    with open(SCHEMA_FILE, "r") as f:
        schema_sql = f.read()
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    print("Schema created.\n")


def load_table(conn, csv_name, table_name, columns=None, dedupe_on=None):
    path = os.path.join(RAW_DIR, csv_name)
    df = pd.read_csv(path)

    if dedupe_on:
        before = len(df)
        df = df.drop_duplicates(subset=dedupe_on)
        after = len(df)
        print(f"  Deduplicated {csv_name} on {dedupe_on}: {before} -> {after} rows")

    if columns:
        df = df[columns]

    # Replace NaN/NaT with None at the value level (more robust across pandas
    # versions than DataFrame.where(), which doesn't reliably convert NaN to
    # None for all column dtypes in pandas 3.x)
    cols = list(df.columns)
    records = []
    for row in df.itertuples(index=False, name=None):
        clean_row = tuple(None if pd.isna(v) else v for v in row)
        records.append(clean_row)

    col_ident = sql.SQL(", ").join(map(sql.Identifier, cols))
    placeholders = sql.SQL(", ").join(sql.Placeholder() * len(cols))
    insert_stmt = sql.SQL("INSERT INTO {table} ({cols}) VALUES ({vals})").format(
        table=sql.Identifier(table_name),
        cols=col_ident,
        vals=placeholders,
    )

    with conn.cursor() as cur:
        cur.executemany(insert_stmt, records)
    conn.commit()
    print(f"  Loaded {len(records)} rows into '{table_name}'")


def main():
    conn = get_connection()
    try:
        run_schema(conn)

        print("Loading lookup tables...")
        load_table(conn, "product_category_name_translation.csv", "category_translation")

        # Patch: 2 category names in products.csv have no row in the translation
        # CSV (found during Phase 1 FK checks). Since category_translation is a
        # FK target for products, insert fallback rows now so the products load
        # doesn't fail. English name falls back to the Portuguese name.
        missing_categories = [
            "pc_gamer",
            "portateis_cozinha_e_preparadores_de_alimentos",
        ]
        with conn.cursor() as cur:
            for cat in missing_categories:
                cur.execute(
                    """
                    INSERT INTO category_translation (product_category_name, product_category_name_english)
                    VALUES (%s, %s)
                    ON CONFLICT (product_category_name) DO NOTHING
                    """,
                    (cat, cat),
                )
        conn.commit()
        print(f"  Added {len(missing_categories)} fallback category translation rows "
              f"(found missing during Phase 1 FK check)")

        load_table(
            conn,
            "olist_geolocation_dataset.csv",
            "geolocation",
            dedupe_on=["geolocation_zip_code_prefix"],
        )

        print("\nLoading entity tables...")
        load_table(conn, "olist_customers_dataset.csv", "customers")
        load_table(conn, "olist_sellers_dataset.csv", "sellers")
        load_table(conn, "olist_products_dataset.csv", "products")

        print("\nLoading transactional tables...")
        load_table(conn, "olist_orders_dataset.csv", "orders")
        load_table(conn, "olist_order_items_dataset.csv", "order_items")
        load_table(conn, "olist_order_payments_dataset.csv", "order_payments")

        # Verify review_id + order_id uniqueness before load, per schema note
        reviews_df = pd.read_csv(os.path.join(RAW_DIR, "olist_order_reviews_dataset.csv"))
        dup_check = reviews_df.duplicated(subset=["review_id", "order_id"]).sum()
        print(f"\nVerifying order_reviews composite key uniqueness (review_id, order_id): "
              f"{dup_check} duplicates found")
        load_table(conn, "olist_order_reviews_dataset.csv", "order_reviews",
                   dedupe_on=["review_id", "order_id"] if dup_check > 0 else None)

        print("\nAll tables loaded successfully.")

    except Exception as e:
        conn.rollback()
        print(f"\nERROR during load: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
