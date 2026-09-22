from typing import Any, Dict, List


class Diagnosis():

    @staticmethod
    def classify_failure(
        result: Dict[str, Any],
        fingerprint: Dict[str, Any],
        baseline_fingerprint: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        """
        Decide whether a failed result is a model-quality failure, an
        environment failure, or inconclusive.
        """
        status = 'pass'
        reasons: List[str] = []
        var_result = {'diagnosis_status': status, 'diagnosis_reasons': reasons}

        task_success = result.get('task_success')
        if task_success is True:
            return var_result

        # Environment check
        env_drift: List[str] = []
        if baseline_fingerprint:
            hw_now = fingerprint.get('hardware', {})
            hw_base = baseline_fingerprint.get('hardware', {})

            if hw_now != hw_base:
                env_drift.append(f'Hadrware set changed: {hw_base} -> {hw_now}')


        # Model check
        model_drift: List[str] = []
        if baseline_fingerprint:
            m_now = fingerprint.get('model')
            m_base = baseline_fingerprint.get('model')

            if m_now != m_base:
                model_drift.append(
                    f'model changed: {m_base} -> {m_now}'
                )

        # Quantization check
        quantization_drift: List[str] = []
        if baseline_fingerprint:
            m_now = fingerprint.get('quantization_level')
            m_base = baseline_fingerprint.get('quantization_level')

            if m_now != m_base:
                quantization_drift.append(
                    f'Quantization changed: {m_base} -> {m_now}'
                )

        # Decision
        if env_drift:
            status = 'environment_failure'
            reasons.extend(env_drift)
        elif baseline_fingerprint is None:
            status = 'inconclusive'
            reasons.append('no baseline fingerprint to compare against')
        else:
            status = 'model_failure'
            reasons.append('fingerprint unchanged; failure likely intrinsic to the model')

        var_result = {'diagnosis_status': status, 'diagnosis_reasons': reasons}
        return var_result
