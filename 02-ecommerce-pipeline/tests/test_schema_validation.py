# tests/test_schema_validation.py
import pytest
import pandas as pd
import pandera as pa
from ingestion.validate_schema import (
    PRODUCTS_SCHEMA, USERS_SCHEMA, CARTS_SCHEMA
)


class TestProductsSchema:
    def valid_products_df(self):
        return pd.DataFrame([
            {"id": 1, "title": "Test Product", "price": 29.99,
             "category": "electronics", "description": "A product"},
            {"id": 2, "title": "Another", "price": 9.99,
             "category": "jewelery",    "description": "Another"},
        ])

    def test_valid_products_passes(self):
        df = self.valid_products_df()
        PRODUCTS_SCHEMA.validate(df)  # should not raise

    def test_negative_price_fails(self):
        df = self.valid_products_df()
        df.loc[0, "price"] = -5.0
        with pytest.raises(pa.errors.SchemaError):
            PRODUCTS_SCHEMA.validate(df)

    def test_zero_price_fails(self):
        df = self.valid_products_df()
        df.loc[0, "price"] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            PRODUCTS_SCHEMA.validate(df)

    def test_null_id_fails(self):
        df = self.valid_products_df()
        df.loc[0, "id"] = None
        df["id"] = df["id"].astype("Int64")
        with pytest.raises(pa.errors.SchemaError):
            PRODUCTS_SCHEMA.validate(df)

    def test_invalid_category_fails(self):
        df = self.valid_products_df()
        df.loc[0, "category"] = "invalid_category"
        with pytest.raises(pa.errors.SchemaError):
            PRODUCTS_SCHEMA.validate(df)

    def test_all_valid_categories_pass(self):
        for cat in ["men's clothing", "women's clothing", "jewelery", "electronics"]:
            df = self.valid_products_df()
            df["category"] = cat
            PRODUCTS_SCHEMA.validate(df)


class TestUsersSchema:
    def valid_users_df(self):
        return pd.DataFrame([
            {"id": 1, "email": "test@test.com", "username": "testuser"},
            {"id": 2, "email": "user2@test.com", "username": "user2"},
        ])

    def test_valid_users_passes(self):
        df = self.valid_users_df()
        USERS_SCHEMA.validate(df)

    def test_null_email_fails(self):
        df = self.valid_users_df()
        df.loc[0, "email"] = None
        with pytest.raises(pa.errors.SchemaError):
            USERS_SCHEMA.validate(df)

    def test_null_username_fails(self):
        df = self.valid_users_df()
        df.loc[0, "username"] = None
        with pytest.raises(pa.errors.SchemaError):
            USERS_SCHEMA.validate(df)


class TestCartsSchema:
    def valid_carts_df(self):
        return pd.DataFrame([
            {"id": 1, "userId": 1, "date": "2024-01-15",
             "products": [{"productId": 1, "quantity": 2}]},
        ])

    def test_valid_carts_passes(self):
        df = self.valid_carts_df()
        CARTS_SCHEMA.validate(df)

    def test_null_date_fails(self):
        df = self.valid_carts_df()
        df.loc[0, "date"] = None
        with pytest.raises(pa.errors.SchemaError):
            CARTS_SCHEMA.validate(df)
