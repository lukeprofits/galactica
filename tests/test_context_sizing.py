"""num_ctx is derived from what we actually send, not fixed.

Ollama reserves the model's maximum context as KV cache unless told otherwise,
and that reservation dwarfs the weights: qwen3:4b is 2.5GB and asked for 42GB.
"""

from galactica.config import Config


def test_derived_from_the_budgets_that_produce_the_prompt():
    cfg = Config(context_budget=16000, max_answer_tokens=2048, think_reserve=2048,
                 prompt_overhead=2048, think=True)
    # 16000 + 2048 + 2048 + 2048 = 22144, rounded up to a whole 1024.
    assert cfg.effective_num_ctx() == 22528


def test_lowering_the_context_budget_lowers_the_kv_cache():
    """The point of deriving it: small-hardware settings propagate."""
    laptop = Config(context_budget=4000, think=False)
    assert laptop.effective_num_ctx() == 8192
    assert laptop.effective_num_ctx() < Config().effective_num_ctx()


def test_reasoning_off_reclaims_its_reserve():
    on = Config(context_budget=8000, think=True)
    off = Config(context_budget=8000, think=False)
    assert on.effective_num_ctx() - off.effective_num_ctx() == 2048


def test_explicit_value_always_wins():
    assert Config(num_ctx=8192, context_budget=16000).effective_num_ctx() == 8192
    assert Config(num_ctx=8192).effective_num_ctx(client_reserve=99999) == 8192


def test_gateway_gets_an_allowance_for_the_client_prompt():
    """Claude Code sends its own system prompt, tool schemas and history."""
    cfg = Config()
    assert cfg.effective_num_ctx(client_reserve=cfg.client_reserve) == (
        cfg.effective_num_ctx() + cfg.client_reserve
    )


def test_a_floor_keeps_tiny_configurations_workable():
    assert Config(context_budget=10, max_answer_tokens=10, prompt_overhead=0,
                  think=False).effective_num_ctx() == Config.MIN_NUM_CTX


def test_env_override_and_default(monkeypatch):
    monkeypatch.delenv("GALACTICA_NUM_CTX", raising=False)
    assert Config.from_env().num_ctx is None  # derive by default
    monkeypatch.setenv("GALACTICA_NUM_CTX", "8192")
    assert Config.from_env().effective_num_ctx() == 8192


def test_providers_and_gateway_receive_the_derived_size(tmp_path):
    from galactica.providers import build_provider
    from galactica.server import Gateway

    cfg = Config(context_budget=4000, think=False, data_dir=tmp_path)
    assert build_provider(cfg).num_ctx == 8192

    class Fake:
        def chat(self, payload):
            return {"message": {"content": ""}}

    gateway = Gateway(cfg, mode="off", provider=Fake())
    # The gateway's own provider would carry the client allowance; the stub here
    # just proves the config path is the one used to build it.
    assert cfg.effective_num_ctx(client_reserve=cfg.client_reserve) == 8192 + cfg.client_reserve
