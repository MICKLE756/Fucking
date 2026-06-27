"""流式输出离线单测：StreamEvent (SSE) / StreamBuffer 背压。"""

import json

from hello_agents.core.streaming import (
    StreamEventType,
    StreamEvent,
    StreamBuffer,
)


def test_stream_event_to_sse_format():
    event = StreamEvent.create(
        StreamEventType.LLM_CHUNK, "agent", text="你好"
    )
    sse = event.to_sse()
    lines = sse.split("\n")
    assert lines[0] == "event: llm_chunk"
    assert lines[1].startswith("data: ")
    # data 行是合法 JSON，且中文未被转义
    payload = json.loads(lines[1][len("data: "):])
    assert payload["type"] == "llm_chunk"
    assert payload["data"]["text"] == "你好"
    # SSE 事件以空行结束
    assert sse.endswith("\n")


def test_stream_buffer_backpressure_drops_oldest():
    buf = StreamBuffer(max_buffer_size=3)
    for i in range(5):
        buf.add(StreamEvent.create(StreamEventType.LLM_CHUNK, "a", idx=i))
    events = buf.get_all()
    assert len(events) == 3
    # 最旧的 0、1 被丢弃，保留 2、3、4
    assert [e.data["idx"] for e in events] == [2, 3, 4]


def test_stream_buffer_filter_and_clear():
    buf = StreamBuffer()
    buf.add(StreamEvent.create(StreamEventType.LLM_CHUNK, "a"))
    buf.add(StreamEvent.create(StreamEventType.ERROR, "a"))
    buf.add(StreamEvent.create(StreamEventType.LLM_CHUNK, "a"))

    chunks = buf.filter_by_type(StreamEventType.LLM_CHUNK)
    assert len(chunks) == 2

    buf.clear()
    assert buf.get_all() == []
