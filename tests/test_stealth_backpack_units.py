import sys
from unittest.mock import MagicMock
import abc

# Mock torch and related modules before importing anything that might use them
mock_torch = MagicMock()
mock_nn = MagicMock()
mock_nn_functional = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_nn
sys.modules['torch.nn.functional'] = mock_nn_functional

# To avoid metaclass conflict: BaseExpert(nn.Module, ABC)
# nn.Module needs to use ABCMeta as its metaclass
class MockNNModule(metaclass=abc.ABCMeta):
    pass

mock_nn.Module = MockNNModule

# Also mock torch.utils.checkpoint as it is used in BaseExpert.forward
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.checkpoint'] = MagicMock()

# Instead of mocking the whole agents package, we can try to mock ONLY base_expert
# before it's imported by stealth_backpack or agents.__init__
sys.modules['hive_zero_core.agents.base_expert'] = MagicMock()

import pytest
from datetime import datetime, UTC

# Now import
from hive_zero_core.agents.stealth_backpack import (
    InfiltrationEngine, CollectionMode, CollectedData
)

class TestInfiltrationEngine:
    """Tests for InfiltrationEngine."""

    @pytest.fixture
    def engine(self):
        """Provide an InfiltrationEngine instance."""
        return InfiltrationEngine(CollectionMode.MOSQUITO)

    def test_initial_state(self, engine):
        """Test that the engine starts with no collected items."""
        assert len(engine.get_all_collected()) == 0
        assert engine.collected_items == []

    def test_clear_collected_empty(self, engine):
        """Test clear_collected on an already empty engine."""
        engine.clear_collected()
        assert len(engine.get_all_collected()) == 0
        assert engine.collected_items == []

    def test_clear_collected_with_items(self, engine):
        """Test that clear_collected removes all items."""
        # Manually add some items
        item1 = CollectedData(
            data_type="test",
            content=b"data1",
            timestamp=datetime.now(UTC),
            source="test_source",
            stealth_score=1.0
        )
        item2 = CollectedData(
            data_type="test",
            content=b"data2",
            timestamp=datetime.now(UTC),
            source="test_source",
            stealth_score=1.0
        )

        engine.collected_items.extend([item1, item2])
        assert len(engine.get_all_collected()) == 2

        # Clear items
        engine.clear_collected()

        # Verify empty
        assert len(engine.get_all_collected()) == 0
        assert engine.collected_items == []

    def test_get_all_collected_returns_copy(self, engine):
        """Test that get_all_collected returns a copy, not the original list."""
        item = CollectedData(
            data_type="test",
            content=b"data",
            timestamp=datetime.now(UTC),
            source="test_source",
            stealth_score=1.0
        )
        engine.collected_items.append(item)

        collected = engine.get_all_collected()
        assert collected == engine.collected_items
        assert collected is not engine.collected_items

        # Modifying the copy shouldn't affect the original
        collected.clear()
        assert len(engine.collected_items) == 1

    def test_collection_methods_populate_list(self, engine):
        """Test that collection methods actually add items to the list."""
        # Test mosquito_collect
        engine.mosquito_collect("target1", ["credentials"])
        assert len(engine.collected_items) == 1

        # Test surgical_collect
        engine.surgical_collect("target1", "/etc/passwd")
        assert len(engine.collected_items) == 2

        # Test passive_collect
        engine.passive_collect({"some": "data"})
        assert len(engine.collected_items) == 3

        # Test vacuum_collect (adds many items)
        engine.vacuum_collect("target2")
        assert len(engine.collected_items) > 3

        # Finally clear everything
        engine.clear_collected()
        assert len(engine.collected_items) == 0
