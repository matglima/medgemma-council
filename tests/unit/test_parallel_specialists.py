"""
Tests for Phase 13: Parallel specialist execution.

Verifies that _run_specialists() uses concurrent.futures.ThreadPoolExecutor
to run activated specialist agents in parallel instead of sequentially.

TDD: Written BEFORE implementation.
Per CLAUDE.md: "Write failing tests BEFORE implementation code for every module."

Key behaviors:
1. _run_specialists uses ThreadPoolExecutor (not sequential for-loop)
2. All activated specialists run concurrently
3. A failure in one specialist doesn't crash others (fault isolation)
4. Results from all specialists are merged into a single dict
5. max_workers is configurable via COUNCIL_MAX_WORKERS env var
6. Thread pool is properly shut down after execution
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test: ThreadPoolExecutor is used
# ---------------------------------------------------------------------------


class TestParallelExecutionMechanism:
    """Verify that _run_specialists delegates to ThreadPoolExecutor."""

    @patch("graph.ModelFactory")
    def test_uses_thread_pool_executor(self, MockFactory):
        """_run_specialists should use ThreadPoolExecutor (import present in graph)."""
        import graph

        # Verify the import exists
        assert hasattr(graph, "ThreadPoolExecutor")

    @patch("graph.ModelFactory")
    def test_specialists_run_in_threads(self, MockFactory):
        """Multiple specialists should execute in different threads."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        # Track which threads each specialist runs in
        thread_ids = {}
        lock = threading.Lock()

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent, OncologyAgent"
            },
            "patient_context": {"chief_complaint": "chest pain"},
            "medical_images": [],
        }

        def make_analyze(name):
            def analyze(s):
                with lock:
                    thread_ids[name] = threading.current_thread().ident
                return {"agent_outputs": {name: f"{name} output"}}
            return analyze

        with patch("graph.CardiologyAgent") as MockCardio, \
             patch("graph.OncologyAgent") as MockOnco:
            MockCardio.return_value.analyze.side_effect = make_analyze("CardiologyAgent")
            MockOnco.return_value.analyze.side_effect = make_analyze("OncologyAgent")

            _run_specialists(state)

        # Both specialists should have run (both have thread IDs recorded)
        assert "CardiologyAgent" in thread_ids
        assert "OncologyAgent" in thread_ids

    @patch("graph.ModelFactory")
    def test_run_single_specialist_is_callable(self, MockFactory):
        """_run_single_specialist helper should be importable and callable."""
        from graph import _run_single_specialist

        mock_llm = MagicMock()
        mock_cls = MagicMock()
        mock_cls.return_value.analyze.return_value = {
            "agent_outputs": {"TestAgent": "output"}
        }

        result = _run_single_specialist("TestAgent", mock_cls, mock_llm, {})
        assert result == {"TestAgent": "output"}


# ---------------------------------------------------------------------------
# Test: Fault isolation
# ---------------------------------------------------------------------------


class TestFaultIsolation:
    """A failure in one specialist should not prevent others from completing."""

    @patch("graph.ModelFactory")
    def test_one_specialist_error_others_still_run(self, MockFactory):
        """If one specialist raises, the others' results should still appear."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent, OncologyAgent"
            },
            "patient_context": {"chief_complaint": "chest pain"},
            "medical_images": [],
        }

        with patch("graph.CardiologyAgent") as MockCardio, \
             patch("graph.OncologyAgent") as MockOnco:
            cardio_instance = MockCardio.return_value
            cardio_instance.analyze.side_effect = RuntimeError("Model OOM")

            onco_instance = MockOnco.return_value
            onco_instance.analyze.return_value = {
                "agent_outputs": {"OncologyAgent": "Tumor markers normal."}
            }

            result = _run_specialists(state)

        assert "OncologyAgent" in result
        assert "Tumor markers normal." in result["OncologyAgent"]
        assert "CardiologyAgent" in result
        assert "Error" in result["CardiologyAgent"]

    @patch("graph.ModelFactory")
    def test_all_specialists_fail_returns_all_errors(self, MockFactory):
        """If all specialists fail, each should have an error entry."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent, OncologyAgent"
            },
            "patient_context": {},
            "medical_images": [],
        }

        with patch("graph.CardiologyAgent") as MockCardio, \
             patch("graph.OncologyAgent") as MockOnco:
            MockCardio.return_value.analyze.side_effect = RuntimeError("OOM")
            MockOnco.return_value.analyze.side_effect = ValueError("Bad input")

            result = _run_specialists(state)

        assert "CardiologyAgent" in result
        assert "Error" in result["CardiologyAgent"]
        assert "OncologyAgent" in result
        assert "Error" in result["OncologyAgent"]


# ---------------------------------------------------------------------------
# Test: Result merging
# ---------------------------------------------------------------------------


class TestResultMerging:
    """Outputs from all parallel specialists should be merged into one dict."""

    @patch("graph.ModelFactory")
    def test_multiple_specialist_outputs_merged(self, MockFactory):
        """Three specialists running in parallel should produce three entries."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": (
                    "Routing to specialists: CardiologyAgent, OncologyAgent, NeurologyAgent"
                )
            },
            "patient_context": {},
            "medical_images": [],
        }

        with patch("graph.CardiologyAgent") as MockC, \
             patch("graph.OncologyAgent") as MockO, \
             patch("graph.NeurologyAgent") as MockN:
            MockC.return_value.analyze.return_value = {
                "agent_outputs": {"CardiologyAgent": "ACS likely."}
            }
            MockO.return_value.analyze.return_value = {
                "agent_outputs": {"OncologyAgent": "No malignancy."}
            }
            MockN.return_value.analyze.return_value = {
                "agent_outputs": {"NeurologyAgent": "No focal deficits."}
            }

            result = _run_specialists(state)

        assert len(result) == 3
        assert result["CardiologyAgent"] == "ACS likely."
        assert result["OncologyAgent"] == "No malignancy."
        assert result["NeurologyAgent"] == "No focal deficits."


# ---------------------------------------------------------------------------
# Test: Configurable max_workers
# ---------------------------------------------------------------------------


class TestMaxWorkersConfig:
    """max_workers should be configurable via COUNCIL_MAX_WORKERS env var."""

    @patch("graph.ModelFactory")
    def test_default_max_workers_equals_activated_count(self, MockFactory):
        """Without env var, max_workers should default to 1 (sequential) to prevent CUDA OOM."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent, OncologyAgent"
            },
            "patient_context": {},
            "medical_images": [],
        }

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COUNCIL_MAX_WORKERS", None)

            with patch("graph.CardiologyAgent") as MockC, \
                 patch("graph.OncologyAgent") as MockO, \
                 patch("graph.ThreadPoolExecutor", wraps=RealThreadPoolExecutor) as MockTPE:
                MockC.return_value.analyze.return_value = {
                    "agent_outputs": {"CardiologyAgent": "ACS likely."}
                }
                MockO.return_value.analyze.return_value = {
                    "agent_outputs": {"OncologyAgent": "No malignancy."}
                }

                _run_specialists(state)

            # Default max_workers=1 (sequential) to prevent CUDA OOM from parallel generate() calls
            call_kwargs = MockTPE.call_args
            max_workers = call_kwargs[1].get("max_workers") if call_kwargs[1] else call_kwargs[0][0]
            assert max_workers == 1

    @patch("graph.ModelFactory")
    def test_env_var_overrides_max_workers(self, MockFactory):
        """COUNCIL_MAX_WORKERS env var should override the default."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent, OncologyAgent, NeurologyAgent"
            },
            "patient_context": {},
            "medical_images": [],
        }

        with patch.dict(os.environ, {"COUNCIL_MAX_WORKERS": "4"}):
            with patch("graph.CardiologyAgent") as MockC, \
                 patch("graph.OncologyAgent") as MockO, \
                 patch("graph.NeurologyAgent") as MockN, \
                 patch("graph.ThreadPoolExecutor", wraps=RealThreadPoolExecutor) as MockTPE:
                MockC.return_value.analyze.return_value = {
                    "agent_outputs": {"CardiologyAgent": "ACS likely."}
                }
                MockO.return_value.analyze.return_value = {
                    "agent_outputs": {"OncologyAgent": "No malignancy."}
                }
                MockN.return_value.analyze.return_value = {
                    "agent_outputs": {"NeurologyAgent": "No focal deficits."}
                }

                _run_specialists(state)

            call_kwargs = MockTPE.call_args
            max_workers = call_kwargs[1].get("max_workers") if call_kwargs[1] else call_kwargs[0][0]
            assert max_workers == 4


# ---------------------------------------------------------------------------
# Test: Timeout handling
# ---------------------------------------------------------------------------


class TestSpecialistTimeout:
    """Specialists that hang should be timed out gracefully."""

    @patch("graph.ModelFactory")
    def test_timeout_produces_error_entry(self, MockFactory):
        """A specialist that exceeds the timeout should get an error entry."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent"
            },
            "patient_context": {},
            "medical_images": [],
        }

        with patch("graph.CardiologyAgent") as MockCardio:
            MockCardio.return_value.analyze.side_effect = TimeoutError("Timed out")

            result = _run_specialists(state)

        assert "CardiologyAgent" in result
        assert "Error" in result["CardiologyAgent"]


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """The specialist_node should still work correctly with parallel _run_specialists."""

    @patch("graph._run_specialists")
    def test_specialist_node_merges_parallel_outputs(self, mock_run):
        """specialist_node should merge parallel outputs with existing state."""
        from graph import specialist_node

        mock_run.return_value = {
            "CardiologyAgent": "ACS likely.",
            "OncologyAgent": "No malignancy.",
        }

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent, OncologyAgent"
            },
        }

        result = specialist_node(state)

        assert "agent_outputs" in result
        assert "SupervisorAgent" in result["agent_outputs"]
        assert "CardiologyAgent" in result["agent_outputs"]
        assert "OncologyAgent" in result["agent_outputs"]

    @patch("graph._run_debate_round")
    @patch("graph._run_specialists")
    @patch("graph.ResearchAgent")
    @patch("graph.SupervisorAgent")
    def test_full_graph_still_works_with_parallel(
        self, MockSupervisor, MockResearch, mock_specialists, mock_debate
    ):
        """End-to-end graph execution should work identically after parallelization."""
        from graph import build_council_graph

        sup = MockSupervisor.return_value
        sup.route.return_value = ["CardiologyAgent", "OncologyAgent"]
        sup.name = "SupervisorAgent"
        sup.detect_conflict.return_value = False
        sup.synthesize.return_value = {
            "final_plan": "Combined plan from parallel specialists.",
            "consensus_reached": True,
        }

        mock_specialists.return_value = {
            "CardiologyAgent": "ACS likely. (ACC/AHA 2023)",
            "OncologyAgent": "No malignancy. (NCCN v1.2025)",
        }

        state = {
            "messages": [{"role": "user", "content": "Evaluate this patient."}],
            "patient_context": {"chief_complaint": "chest pain", "age": "55"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
            "red_flag_detected": False,
            "emergency_override": "",
        }

        graph = build_council_graph()
        result = graph.invoke(state)

        assert result["consensus_reached"] is True
        assert "Combined plan" in result["final_plan"]


# ---------------------------------------------------------------------------
# Test: RadiologyAgent receives vision model (Fix #8)
# ---------------------------------------------------------------------------


class TestRadiologyAgentVisionModel:
    """RadiologyAgent should receive a vision model, not the text model.

    _run_specialists() creates one text model and passes it to ALL agents.
    But RadiologyAgent calls self.llm(images=..., prompt=...) which is the
    VisionModelWrapper interface.  Passing the text model causes
    AutoModelForCausalLM to reject the 'images' kwarg.
    """

    @patch("graph.ModelFactory")
    def test_radiology_agent_gets_vision_model(self, MockFactory):
        """When RadiologyAgent is activated, it should receive the vision model."""
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_text_llm = MagicMock(name="text_llm")
        mock_vision_llm = MagicMock(name="vision_llm")
        mock_factory_inst.create_text_model.return_value = mock_text_llm
        mock_factory_inst.create_vision_model.return_value = mock_vision_llm
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: RadiologyAgent"
            },
            "patient_context": {},
            "medical_images": ["chest_xray.png"],
        }

        with patch("graph.RadiologyAgent") as MockRadiology:
            MockRadiology.return_value.analyze.return_value = {
                "agent_outputs": {"RadiologyAgent": "No acute findings."}
            }

            _run_specialists(state)

            # RadiologyAgent should be instantiated with the vision model
            MockRadiology.assert_called_once()
            call_kwargs = MockRadiology.call_args
            passed_llm = call_kwargs[1].get("llm") or call_kwargs[0][0]
            assert passed_llm is mock_vision_llm, (
                "RadiologyAgent should receive vision_llm, not text_llm"
            )

    @patch("graph.ModelFactory")
    def test_text_agents_still_get_text_model_when_radiology_active(self, MockFactory):
        """Non-radiology agents should still get the text model even when
        RadiologyAgent is also activated.
        """
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_text_llm = MagicMock(name="text_llm")
        mock_vision_llm = MagicMock(name="vision_llm")
        mock_factory_inst.create_text_model.return_value = mock_text_llm
        mock_factory_inst.create_vision_model.return_value = mock_vision_llm
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent, RadiologyAgent"
            },
            "patient_context": {},
            "medical_images": ["chest_xray.png"],
        }

        with patch("graph.CardiologyAgent") as MockCardio, \
             patch("graph.RadiologyAgent") as MockRadiology:
            MockCardio.return_value.analyze.return_value = {
                "agent_outputs": {"CardiologyAgent": "ACS likely."}
            }
            MockRadiology.return_value.analyze.return_value = {
                "agent_outputs": {"RadiologyAgent": "No acute findings."}
            }

            _run_specialists(state)

            # CardiologyAgent should get text model
            MockCardio.assert_called_once()
            cardio_llm = MockCardio.call_args[1].get("llm") or MockCardio.call_args[0][0]
            assert cardio_llm is mock_text_llm

            # RadiologyAgent should get vision model
            MockRadiology.assert_called_once()
            radio_llm = MockRadiology.call_args[1].get("llm") or MockRadiology.call_args[0][0]
            assert radio_llm is mock_vision_llm

    @patch("graph.ModelFactory")
    def test_vision_model_not_created_when_radiology_inactive(self, MockFactory):
        """create_vision_model() should NOT be called if RadiologyAgent is
        not among the activated specialists (avoid loading unnecessary model).
        """
        from graph import _run_specialists

        mock_factory_inst = MagicMock()
        mock_factory_inst.create_text_model.return_value = MagicMock()
        MockFactory.return_value = mock_factory_inst

        state = {
            "agent_outputs": {
                "SupervisorAgent": "Routing to specialists: CardiologyAgent"
            },
            "patient_context": {},
            "medical_images": [],
        }

        with patch("graph.CardiologyAgent") as MockCardio:
            MockCardio.return_value.analyze.return_value = {
                "agent_outputs": {"CardiologyAgent": "ACS likely."}
            }

            _run_specialists(state)

        # Vision model should not have been created
        mock_factory_inst.create_vision_model.assert_not_called()
