import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The real SDK imports torch. Unit tests never run inference, so a placeholder
# type is enough to import the decoder and the caller.
if "llm_sdk" not in sys.modules:
    llm_sdk = types.ModuleType("llm_sdk")

    class Small_LLM_Model:
        """Placeholder used only to satisfy type imports in tests."""

    llm_sdk.Small_LLM_Model = Small_LLM_Model
    sys.modules["llm_sdk"] = llm_sdk
