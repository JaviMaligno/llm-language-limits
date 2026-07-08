from llm_language_limits.clients.base import ChatResult, FakeClient

def test_fakeclient_echoes_last_user_message_by_default():
    c = FakeClient()
    res = c.chat([{"role": "user", "content": "ping"}], system="s",
                 temperature=0.0, max_tokens=64)
    assert isinstance(res, ChatResult)
    assert "ping" in res.text
    assert res.output_tokens > 0

def test_fakeclient_uses_custom_reply_fn():
    c = FakeClient(reply_fn=lambda msgs: "FIXED")
    res = c.chat([{"role": "user", "content": "x"}], system="s",
                 temperature=0.0, max_tokens=64)
    assert res.text == "FIXED"
