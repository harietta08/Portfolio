# tests/test_ab_test_stats.py
import pytest
import numpy as np
from analysis.ab_test_stats import (
    calculate_sample_size,
    run_ab_test,
    check_guardrail_metric,
    generate_recommendation
)


class TestSampleSize:
    def test_returns_required_keys(self):
        result = calculate_sample_size()
        required = ["sample_size_per_variant", "total_sample_size",
                    "baseline_rate", "mde_absolute", "alpha", "power"]
        for key in required:
            assert key in result

    def test_sample_size_positive(self):
        result = calculate_sample_size()
        assert result["sample_size_per_variant"] > 0

    def test_total_is_double_per_variant(self):
        result = calculate_sample_size()
        assert result["total_sample_size"] == result["sample_size_per_variant"] * 2

    def test_higher_power_needs_more_samples(self):
        low_power  = calculate_sample_size(power=0.80)
        high_power = calculate_sample_size(power=0.90)
        assert high_power["sample_size_per_variant"] > low_power["sample_size_per_variant"]

    def test_smaller_mde_needs_more_samples(self):
        large_mde = calculate_sample_size(mde=0.05)
        small_mde = calculate_sample_size(mde=0.01)
        assert small_mde["sample_size_per_variant"] > large_mde["sample_size_per_variant"]


class TestRunAbTest:
    def test_significant_result(self):
        # Large sample, clear difference — should be significant
        result = run_ab_test(
            control_conversions=380,   control_sessions=1000,
            treatment_conversions=450, treatment_sessions=1000
        )
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_insignificant_result(self):
        # Tiny sample, tiny difference — should NOT be significant
        result = run_ab_test(
            control_conversions=38, control_sessions=100,
            treatment_conversions=40, treatment_sessions=100
        )
        assert result["significant"] is False

    def test_lift_calculation(self):
        result = run_ab_test(
            control_conversions=38,  control_sessions=100,
            treatment_conversions=41, treatment_sessions=100
        )
        assert abs(result["lift_absolute_pp"] - 3.0) < 0.01

    def test_ci_bounds_exist(self):
        result = run_ab_test(
            control_conversions=380,   control_sessions=1000,
            treatment_conversions=410, treatment_sessions=1000
        )
        assert "ci_lower_pp" in result
        assert "ci_upper_pp" in result

    def test_returns_all_required_keys(self):
        result = run_ab_test(380, 1000, 410, 1000)
        for key in ["control_rate", "treatment_rate", "z_statistic",
                    "p_value", "significant", "lift_absolute_pp",
                    "ci_lower_pp", "ci_upper_pp"]:
            assert key in result


class TestGuardrailMetric:
    def test_no_change_passes_guardrail(self):
        np.random.seed(42)
        control   = np.random.normal(55, 10, 500)
        treatment = np.random.normal(55, 10, 500)
        result    = check_guardrail_metric(control, treatment, "aov")
        assert result["guardrail_passed"] is True

    def test_large_drop_fails_guardrail(self):
        np.random.seed(42)
        control   = np.random.normal(100, 5, 1000)
        treatment = np.random.normal(70,  5, 1000)  # 30% drop
        result    = check_guardrail_metric(control, treatment, "aov")
        assert result["guardrail_passed"] is False

    def test_returns_both_means(self):
        control   = np.array([50.0, 55.0, 60.0])
        treatment = np.array([48.0, 53.0, 58.0])
        result    = check_guardrail_metric(control, treatment, "aov")
        assert "control_mean" in result
        assert "treatment_mean" in result


class TestGenerateRecommendation:
    def sample_size_info(self):
        return calculate_sample_size()

    def test_rollout_when_significant_and_guardrails_pass(self):
        test_results = {
            "significant": True, "lift_absolute_pp": 3.0,
            "ci_string": "(1.0pp, 5.0pp)", "p_value": 0.01,
            "control_sessions": 10000, "treatment_sessions": 10000
        }
        guardrails = [{"guardrail_passed": True, "metric_name": "aov"}]
        rec = generate_recommendation(test_results, guardrails, self.sample_size_info())
        assert "ROLLOUT" in rec

    def test_no_rollout_when_guardrail_fails(self):
        test_results = {
            "significant": True, "lift_absolute_pp": 3.0,
            "ci_string": "(1.0pp, 5.0pp)", "p_value": 0.01,
            "control_sessions": 10000, "treatment_sessions": 10000
        }
        guardrails = [{"guardrail_passed": False, "metric_name": "aov"}]
        rec = generate_recommendation(test_results, guardrails, self.sample_size_info())
        assert "GUARDRAIL" in rec

    def test_continue_when_not_significant(self):
        test_results = {
            "significant": False, "lift_absolute_pp": 0.5,
            "ci_string": "(-1.0pp, 2.0pp)", "p_value": 0.30,
            "control_sessions": 100, "treatment_sessions": 100
        }
        guardrails = [{"guardrail_passed": True, "metric_name": "aov"}]
        rec = generate_recommendation(test_results, guardrails, self.sample_size_info())
        assert "CONTINUE" in rec or "INSUFFICIENT" in rec
