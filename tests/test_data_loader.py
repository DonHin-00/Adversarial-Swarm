"""
Tests for data loading utilities.
"""

from hive_zero_core.training.data_loader import DataConfig, NetworkLogDataset


def test_network_log_dataset_synthetic():
    """Test creating a synthetic dataset."""
    dataset = NetworkLogDataset(
        synthetic=True,
        num_synthetic_samples=100,
        batch_size=10,
    )

    assert len(dataset) == 100
    assert dataset.num_batches == 10


def test_network_log_dataset_from_list():
    """Test creating dataset from a list of logs."""
    logs = [
        {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": 80, "proto": 6},
        {"src_ip": "192.168.1.2", "dst_ip": "10.0.0.6", "port": 443, "proto": 6},
    ]

    dataset = NetworkLogDataset(data_source=logs, batch_size=1)

    assert len(dataset) == 2
    assert dataset.num_batches == 2


def test_network_log_dataset_iteration():
    """Test iterating over dataset batches."""
    dataset = NetworkLogDataset(
        synthetic=True,
        num_synthetic_samples=50,
        batch_size=10,
    )

    batch_count = 0
    for batch in dataset:
        batch_count += 1
        assert isinstance(batch, list)
        assert len(batch) <= 10

    assert batch_count == 5


def test_network_log_dataset_get_batch():
    """Test getting a specific batch."""
    dataset = NetworkLogDataset(
        synthetic=True,
        num_synthetic_samples=30,
        batch_size=10,
    )

    # Get first batch
    batch_0 = dataset.get_batch(0)
    assert len(batch_0) == 10

    # Get second batch
    batch_1 = dataset.get_batch(1)
    assert len(batch_1) == 10

    # Get last batch (may be smaller)
    batch_2 = dataset.get_batch(2)
    assert len(batch_2) == 10


def test_synthetic_log_structure():
    """Test that synthetic logs have the correct structure."""
    dataset = NetworkLogDataset(
        synthetic=True,
        num_synthetic_samples=10,
        batch_size=10,
    )

    batch = dataset.get_batch(0)

    for log in batch:
        # Check required fields
        assert "timestamp" in log
        assert "event" in log
        assert "src_ip" in log
        assert "dst_ip" in log
        assert "port" in log
        assert "proto" in log
        assert "bytes" in log
        assert "flags" in log

        # Check types
        assert isinstance(log["src_ip"], str)
        assert isinstance(log["dst_ip"], str)
        assert isinstance(log["port"], int)
        assert isinstance(log["proto"], int)


def test_data_config_integration():
    """Test DataConfig integration with NetworkLogDataset."""
    config = DataConfig(
        batch_size=16,
        synthetic=True,
        num_synthetic_samples=100,
    )

    dataset = config.create_dataset()

    assert len(dataset) == 100
    assert dataset.batch_size == 16


def test_dataset_empty():
    """Test dataset with no data."""
    dataset = NetworkLogDataset(
        data_source=[],
        batch_size=10,
    )

    assert len(dataset) == 0
    assert dataset.num_batches == 0


def test_dataset_single_item():
    """Test dataset with single item."""
    dataset = NetworkLogDataset(
        data_source=[{"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5"}],
        batch_size=10,
    )

    assert len(dataset) == 1
    assert dataset.num_batches == 1

    batch = dataset.get_batch(0)
    assert len(batch) == 1


def test_dataset_batch_size_larger_than_data():
    """Test dataset where batch size is larger than total data."""
    dataset = NetworkLogDataset(
        synthetic=True,
        num_synthetic_samples=5,
        batch_size=10,
    )

    assert len(dataset) == 5
    assert dataset.num_batches == 1

    batch = dataset.get_batch(0)
    assert len(batch) == 5


def test_load_json_file(tmp_path):
    """Test loading logs from JSON file."""
    import json

    # Create a test JSON file
    json_file = tmp_path / "test_logs.json"
    logs_data = [
        {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": 80, "proto": 6},
        {"src_ip": "192.168.1.2", "dst_ip": "10.0.0.6", "port": 443, "proto": 6},
    ]
    with open(json_file, 'w') as f:
        json.dump(logs_data, f)

    dataset = NetworkLogDataset(data_source=json_file, batch_size=10)

    assert len(dataset) == 2
    assert dataset.data[0]["src_ip"] == "192.168.1.1"


def test_load_json_file_with_wrapper(tmp_path):
    """Test loading logs from JSON file with 'logs' wrapper."""
    import json

    # Create a test JSON file with wrapper
    json_file = tmp_path / "test_logs_wrapped.json"
    logs_data = {
        "logs": [
            {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": 80, "proto": 6},
            {"src_ip": "192.168.1.2", "dst_ip": "10.0.0.6", "port": 443, "proto": 6},
        ]
    }
    with open(json_file, 'w') as f:
        json.dump(logs_data, f)

    dataset = NetworkLogDataset(data_source=json_file, batch_size=10)

    assert len(dataset) == 2


def test_load_jsonl_file(tmp_path):
    """Test loading logs from JSONL file."""
    import json

    # Create a test JSONL file
    jsonl_file = tmp_path / "test_logs.jsonl"
    with open(jsonl_file, 'w') as f:
        f.write(json.dumps({"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": 80, "proto": 6}) + "\n")
        f.write(json.dumps({"src_ip": "192.168.1.2", "dst_ip": "10.0.0.6", "port": 443, "proto": 6}) + "\n")

    dataset = NetworkLogDataset(data_source=jsonl_file, batch_size=10)

    assert len(dataset) == 2


def test_load_csv_file(tmp_path):
    """Test loading logs from CSV file."""
    import csv

    # Create a test CSV file
    csv_file = tmp_path / "test_logs.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["src_ip", "dst_ip", "port", "proto"])
        writer.writeheader()
        writer.writerow({"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": "80", "proto": "6"})
        writer.writerow({"src_ip": "192.168.1.2", "dst_ip": "10.0.0.6", "port": "443", "proto": "6"})

    dataset = NetworkLogDataset(data_source=csv_file, batch_size=10)

    assert len(dataset) == 2


def test_load_nonexistent_file():
    """Test loading from a file that doesn't exist falls back to synthetic."""
    dataset = NetworkLogDataset(
        data_source="/nonexistent/path/to/file.json",
        batch_size=10,
        num_synthetic_samples=50,
    )

    # Should fall back to synthetic data
    assert len(dataset) == 50


def test_load_unsupported_format(tmp_path):
    """Test loading from unsupported format falls back to synthetic."""
    txt_file = tmp_path / "test_logs.txt"
    txt_file.write_text("some text data")

    dataset = NetworkLogDataset(
        data_source=txt_file,
        batch_size=10,
        num_synthetic_samples=50,
    )

    # Should fall back to synthetic data
    assert len(dataset) == 50
