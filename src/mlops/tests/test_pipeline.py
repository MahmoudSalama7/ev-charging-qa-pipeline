import pytest
from src.orchestration.workflow import EVQAWorkflow
from src.data_collection.pdf_extractor import PDFExtractor

class TestPipeline:
    @pytest.fixture
    def workflow(self):
        return EVQAWorkflow()

    def test_pipeline_execution(self, workflow):
        assert workflow.run_pipeline() is True

    def test_concurrent_runs(self, workflow):
        # Test that pipeline can't run multiple times concurrently
        workflow.is_running = True
        assert workflow.run_pipeline() is False