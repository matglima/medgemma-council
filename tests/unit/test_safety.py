"""
Tests for safety utilities: red flag scanning and PII redaction.

TDD: Written BEFORE src/utils/safety.py.
Per CLAUDE.md: Every agent output must be scanned for 'Red Flags'
(suicide risk, sepsis shock). If found, override with emergency referral.
"""

import pytest


class TestRedFlagScanner:
    """Tests for the red flag detection system."""

    def test_detects_suicide_risk(self):
        """Must detect suicide-related language and flag it."""
        from utils.safety import scan_for_red_flags

        text = "Patient expresses suicidal ideation and has a plan."
        result = scan_for_red_flags(text)
        assert result["flagged"] is True
        assert "suicide" in result["flags"][0].lower()

    def test_detects_sepsis_indicators(self):
        """Must detect sepsis/septic shock language."""
        from utils.safety import scan_for_red_flags

        text = "Patient presenting with septic shock, lactate 6.2, MAP 55."
        result = scan_for_red_flags(text)
        assert result["flagged"] is True
        assert any("sepsis" in f.lower() or "septic" in f.lower() for f in result["flags"])

    def test_detects_anaphylaxis(self):
        """Must detect anaphylaxis emergency."""
        from utils.safety import scan_for_red_flags

        text = "Acute anaphylaxis after penicillin administration."
        result = scan_for_red_flags(text)
        assert result["flagged"] is True

    def test_no_flags_on_normal_text(self):
        """Normal clinical text should not be flagged."""
        from utils.safety import scan_for_red_flags

        text = "Patient reports mild headache, improved with acetaminophen."
        result = scan_for_red_flags(text)
        assert result["flagged"] is False
        assert result["flags"] == []

    def test_returns_emergency_referral_message(self):
        """When flagged, must include an emergency referral override message."""
        from utils.safety import scan_for_red_flags

        text = "Patient is actively suicidal."
        result = scan_for_red_flags(text)
        assert result["flagged"] is True
        assert "emergency_message" in result
        assert len(result["emergency_message"]) > 0

    def test_detects_cardiac_arrest(self):
        """Must detect cardiac arrest indicators."""
        from utils.safety import scan_for_red_flags

        text = "Patient found unresponsive, no pulse. Cardiac arrest suspected."
        result = scan_for_red_flags(text)
        assert result["flagged"] is True

    def test_detects_stroke_indicators(self):
        """Must detect acute stroke language."""
        from utils.safety import scan_for_red_flags

        text = "Acute ischemic stroke with left-sided hemiplegia, onset 45 min ago."
        result = scan_for_red_flags(text)
        assert result["flagged"] is True


class TestPIIRedaction:
    """Tests for Protected Health Information redaction."""

    def test_redacts_ssn(self):
        """Must redact Social Security Numbers."""
        from utils.safety import redact_pii

        text = "Patient SSN: 123-45-6789."
        result = redact_pii(text)
        assert "123-45-6789" not in result
        assert "[REDACTED_SSN]" in result

    def test_redacts_phone_number(self):
        """Must redact phone numbers."""
        from utils.safety import redact_pii

        text = "Contact: (555) 123-4567."
        result = redact_pii(text)
        assert "(555) 123-4567" not in result
        assert "[REDACTED_PHONE]" in result

    def test_redacts_email(self):
        """Must redact email addresses."""
        from utils.safety import redact_pii

        text = "Email the patient at john.doe@hospital.com for follow-up."
        result = redact_pii(text)
        assert "john.doe@hospital.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_preserves_clinical_content(self):
        """Must not alter clinical content that is not PII."""
        from utils.safety import redact_pii

        text = "Troponin 0.08 ng/mL. BP 120/80. Aspirin 81mg daily."
        result = redact_pii(text)
        assert result == text

    def test_redacts_mrn(self):
        """Must redact Medical Record Numbers (MRN: followed by digits)."""
        from utils.safety import redact_pii

        text = "MRN: 12345678. Patient presents with chest pain."
        result = redact_pii(text)
        assert "12345678" not in result
        assert "[REDACTED_MRN]" in result


class TestDisclaimer:
    """Tests for the clinical disclaimer system."""

    def test_disclaimer_is_appended(self):
        """All outputs must include a clinical disclaimer."""
        from utils.safety import add_disclaimer

        output = "Recommend starting aspirin 81mg daily."
        result = add_disclaimer(output)
        assert "not a substitute" in result.lower() or "disclaimer" in result.lower()
        assert output in result
