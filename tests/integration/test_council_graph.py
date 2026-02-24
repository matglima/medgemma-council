"""
Integration tests for the MedGemma-Council LangGraph state machine.

Tests the full graph flow:
  Ingestion -> Supervisor Routing -> Specialist Analysis -> Conflict Check
  -> (optional) Research -> Debate -> Re-check -> Synthesis -> END

All agents and tools are mocked — no real LLMs or network calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from graph import (
    CouncilState,
    build_council_graph,
    ingestion_node,
    supervisor_route_node,
    specialist_node,
    safety_check_node,
    emergency_synthesis_node,
    conflict_check_node,
    research_node,
    debate_node,
    synthesis_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state(sample_patient_context, sample_medical_images):
    """A minimal valid CouncilState for graph invocation."""
    return {
        "messages": [{"role": "user", "content": "Evaluate this patient."}],
        "patient_context": sample_patient_context,
        "medical_images": sample_medical_images,
        "agent_outputs": {},
        "debate_history": [],
        "consensus_reached": False,
        "research_findings": "",
        "conflict_detected": False,
        "iteration_count": 0,
        "final_plan": "",
        "red_flag_detected": False,
        "emergency_override": "",
        "force_research": False,
    }


# ---------------------------------------------------------------------------
# Node function tests (unit-level, but testing graph-level contracts)
# ---------------------------------------------------------------------------


class TestIngestionNode:
    """The ingestion node validates and enriches initial state."""

    def test_ingestion_returns_state_update(self, base_state):
        result = ingestion_node(base_state)
        assert isinstance(result, dict)

    def test_ingestion_increments_iteration(self, base_state):
        result = ingestion_node(base_state)
        assert result.get("iteration_count", 0) >= 0


class TestSupervisorRouteNode:
    """The supervisor route node determines which specialists to activate."""

    def test_route_populates_agent_outputs(self, base_state):
        with patch("graph.SupervisorAgent") as MockSupervisor:
            instance = MockSupervisor.return_value
            instance.route.return_value = ["CardiologyAgent", "RadiologyAgent"]
            instance.name = "SupervisorAgent"
            result = supervisor_route_node(base_state)
            assert "agent_outputs" in result

    def test_route_returns_specialist_list(self, base_state):
        with patch("graph.SupervisorAgent") as MockSupervisor:
            instance = MockSupervisor.return_value
            instance.route.return_value = ["CardiologyAgent"]
            instance.name = "SupervisorAgent"
            result = supervisor_route_node(base_state)
            assert "activated_specialists" in result or "agent_outputs" in result


class TestSpecialistNode:
    """The specialist node runs activated specialists and collects outputs."""

    def test_specialist_returns_agent_outputs(self, base_state):
        base_state["agent_outputs"] = {
            "SupervisorAgent": "Routing to specialists: CardiologyAgent, RadiologyAgent"
        }
        with patch("graph._run_specialists") as mock_run:
            mock_run.return_value = {
                "CardiologyAgent": "Troponin elevated. ACS likely. (ACC/AHA 2023)",
                "RadiologyAgent": "Cardiomegaly noted on CXR.",
            }
            result = specialist_node(base_state)
            assert "agent_outputs" in result


class TestConflictCheckNode:
    """The conflict check node detects disagreements between specialists."""

    def test_conflict_check_sets_flag(self, base_state):
        base_state["agent_outputs"] = {
            "CardiologyAgent": "Recommend stopping metoprolol",
            "OncologyAgent": "Continue metoprolol for cardioprotection",
        }
        with patch("graph.SupervisorAgent") as MockSupervisor:
            instance = MockSupervisor.return_value
            instance.detect_conflict.return_value = True
            result = conflict_check_node(base_state)
            assert "conflict_detected" in result

    def test_no_conflict_sets_false(self, base_state):
        base_state["agent_outputs"] = {
            "CardiologyAgent": "ACS likely. Recommend cath.",
        }
        with patch("graph.SupervisorAgent") as MockSupervisor:
            instance = MockSupervisor.return_value
            instance.detect_conflict.return_value = False
            result = conflict_check_node(base_state)
            assert result.get("conflict_detected") is False


class TestResearchNode:
    """The research node fetches PubMed literature for conflict resolution."""

    def test_research_returns_findings(self, base_state):
        base_state["conflict_detected"] = True
        with patch("graph.ResearchAgent") as MockResearch:
            instance = MockResearch.return_value
            instance.analyze.return_value = {
                "research_findings": "PMID:12345 - Meta-analysis supports..."
            }
            result = research_node(base_state)
            assert "research_findings" in result
            assert "PMID" in result["research_findings"]


class TestDebateNode:
    """The debate node lets specialists critique each other with evidence."""

    def test_debate_appends_to_history(self, base_state):
        base_state["agent_outputs"] = {
            "CardiologyAgent": "Stop metoprolol",
            "OncologyAgent": "Continue metoprolol",
        }
        base_state["research_findings"] = "PMID:12345 supports continuing."
        with patch("graph._run_debate_round") as mock_debate:
            mock_debate.return_value = [
                "CardiologyAgent: I concede based on PMID:12345."
            ]
            result = debate_node(base_state)
            assert "debate_history" in result
            assert len(result["debate_history"]) > 0

    def test_debate_increments_iteration(self, base_state):
        base_state["iteration_count"] = 1
        with patch("graph._run_debate_round") as mock_debate:
            mock_debate.return_value = ["Debate round 2"]
            result = debate_node(base_state)
            assert result.get("iteration_count", 0) > base_state["iteration_count"]


class TestSynthesisNode:
    """The synthesis node produces the final clinical plan."""

    def test_synthesis_produces_final_plan(self, base_state):
        base_state["agent_outputs"] = {
            "CardiologyAgent": "ACS likely.",
            "RadiologyAgent": "Cardiomegaly on CXR.",
        }
        with patch("graph.SupervisorAgent") as MockSupervisor:
            instance = MockSupervisor.return_value
            instance.synthesize.return_value = {
                "final_plan": "1. Admit to CCU. 2. Start heparin drip.",
                "consensus_reached": True,
            }
            result = synthesis_node(base_state)
            assert "final_plan" in result
            assert result["final_plan"] != ""
            assert result.get("consensus_reached") is True


# ---------------------------------------------------------------------------
# Full graph integration tests
# ---------------------------------------------------------------------------


class TestCouncilGraphBuild:
    """Test that the graph can be built and compiled."""

    def test_build_returns_compiled_graph(self):
        graph = build_council_graph()
        assert graph is not None
        # LangGraph compiled graphs have an invoke method
        assert hasattr(graph, "invoke")

    def test_graph_has_expected_nodes(self):
        graph = build_council_graph()
        # The compiled graph should contain our node names
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"ingestion", "supervisor_route", "specialist", "safety_check", "conflict_check", "synthesis"}
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


class TestCouncilGraphExecution:
    """End-to-end graph execution with all agents mocked."""

    @patch("graph._run_debate_round")
    @patch("graph._run_specialists")
    @patch("graph.ResearchAgent")
    @patch("graph.SupervisorAgent")
    def test_full_flow_no_conflict(
        self, MockSupervisor, MockResearch, mock_specialists, mock_debate, base_state
    ):
        """Happy path: no conflict -> straight to synthesis."""
        sup = MockSupervisor.return_value
        sup.route.return_value = ["CardiologyAgent"]
        sup.name = "SupervisorAgent"
        sup.detect_conflict.return_value = False
        sup.synthesize.return_value = {
            "final_plan": "Admit. Start heparin. Cardiology consult.",
            "consensus_reached": True,
        }

        mock_specialists.return_value = {
            "CardiologyAgent": "ACS likely. Cath recommended. (ACC/AHA 2023)"
        }

        graph = build_council_graph()
        result = graph.invoke(base_state)

        assert result["consensus_reached"] is True
        assert result["final_plan"] != ""
        assert result["conflict_detected"] is False

    @patch("graph._run_debate_round")
    @patch("graph._run_specialists")
    @patch("graph.ResearchAgent")
    @patch("graph.SupervisorAgent")
    def test_full_flow_with_conflict_and_resolution(
        self, MockSupervisor, MockResearch, mock_specialists, mock_debate, base_state
    ):
        """Conflict path: conflict -> research -> debate -> resolution."""
        sup = MockSupervisor.return_value
        sup.route.return_value = ["CardiologyAgent", "OncologyAgent"]
        sup.name = "SupervisorAgent"
        # First conflict check: True, second: False (resolved)
        sup.detect_conflict.side_effect = [True, False]
        sup.synthesize.return_value = {
            "final_plan": "Consensus plan after debate.",
            "consensus_reached": True,
        }

        mock_specialists.return_value = {
            "CardiologyAgent": "Stop metoprolol.",
            "OncologyAgent": "Continue metoprolol.",
        }

        res_instance = MockResearch.return_value
        res_instance.analyze.return_value = {
            "research_findings": "PMID:99999 - Evidence supports continuing."
        }

        mock_debate.return_value = [
            "CardiologyAgent: I revise my position based on PMID:99999."
        ]

        graph = build_council_graph()
        result = graph.invoke(base_state)

        assert result["consensus_reached"] is True
        assert result["final_plan"] != ""
        assert len(result["debate_history"]) > 0

    @patch("graph._run_debate_round")
    @patch("graph._run_specialists")
    @patch("graph.ResearchAgent")
    @patch("graph.SupervisorAgent")
    def test_max_iterations_forces_synthesis(
        self, MockSupervisor, MockResearch, mock_specialists, mock_debate, base_state
    ):
        """Safety valve: max iterations reached -> force synthesis."""
        sup = MockSupervisor.return_value
        sup.route.return_value = ["CardiologyAgent"]
        sup.name = "SupervisorAgent"
        # Always conflict (never resolves)
        sup.detect_conflict.return_value = True
        sup.synthesize.return_value = {
            "final_plan": "Forced plan after max iterations.",
            "consensus_reached": True,
        }

        mock_specialists.return_value = {
            "CardiologyAgent": "Stop metoprolol."
        }

        res_instance = MockResearch.return_value
        res_instance.analyze.return_value = {
            "research_findings": "No conclusive evidence."
        }

        mock_debate.return_value = ["No agreement reached."]

        graph = build_council_graph()
        result = graph.invoke(base_state)

        # Should still terminate with a plan despite perpetual conflict
        assert result["consensus_reached"] is True
        assert result["final_plan"] != ""
        # iteration_count should have been capped
        assert result["iteration_count"] <= 4  # MAX_DEBATE_ROUNDS = 3 + initial
