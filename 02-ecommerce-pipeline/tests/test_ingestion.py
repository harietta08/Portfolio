# tests/test_ingestion.py
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from ingestion.api_client import fetch_endpoint, save_raw, run_ingestion

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"


def mock_products_response():
    return [
        {"id": 1, "title": "Test Product", "price": 29.99,
         "category": "electronics", "rating": {"rate": 4.0, "count": 100}},
        {"id": 2, "title": "Another Product", "price": 9.99,
         "category": "jewelery", "rating": {"rate": 3.5, "count": 50}},
    ]


def mock_users_response():
    return [
        {"id": 1, "email": "test@test.com", "username": "testuser",
         "password": "secret123", "name": {"firstname": "John", "lastname": "Doe"},
         "address": {"city": "Chicago", "zipcode": "60601"}},
    ]


def mock_carts_response():
    return [
        {"id": 1, "userId": 1, "date": "2024-01-15",
         "products": [{"productId": 1, "quantity": 2}]},
    ]


class TestFetchEndpoint:
    def test_fetch_returns_list(self):
        with patch("ingestion.api_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_products_response()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            result = fetch_endpoint("products")
            assert isinstance(result, list)
            assert len(result) == 2

    def test_fetch_raises_on_error(self):
        with patch("ingestion.api_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("526 SSL Error")
            mock_get.return_value = mock_resp

            with pytest.raises(Exception):
                fetch_endpoint("products")

    def test_fetch_correct_url(self):
        with patch("ingestion.api_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = []
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            fetch_endpoint("products")
            call_url = mock_get.call_args[0][0]
            assert "fakestoreapi.com" in call_url or "dummyjson.com" in call_url
            assert "products" in call_url


class TestSaveRaw:
    def test_save_creates_file(self, tmp_path):
        with patch("ingestion.api_client.RAW_DATA_DIR", tmp_path):
            data = mock_products_response()
            path = save_raw(data, "products")
            assert Path(path).exists()

    def test_save_valid_json(self, tmp_path):
        with patch("ingestion.api_client.RAW_DATA_DIR", tmp_path):
            data = mock_products_response()
            path = save_raw(data, "products")
            with open(path) as f:
                loaded = json.load(f)
            assert len(loaded) == len(data)
            assert loaded[0]["id"] == 1

    def test_save_correct_record_count(self, tmp_path):
        with patch("ingestion.api_client.RAW_DATA_DIR", tmp_path):
            data = mock_products_response()
            path = save_raw(data, "products")
            with open(path) as f:
                loaded = json.load(f)
            assert len(loaded) == 2
