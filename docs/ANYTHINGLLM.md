# Optional AnythingLLM integration notes

AnythingLLM is not a dependency. No application storage, database, private provider patch, or compatibility service is bundled.

For an OpenAI-compatible client, point its llama.cpp provider at a reachable `/v1` endpoint. Loopback binding is the default; a container has a different loopback namespace. Configure networking and access controls deliberately instead of exposing an unauthenticated server by default.

Generic lessons from the integration work:

1. Preserve true streaming. An ordinary-text proxy should request upstream streaming and relay each SSE event as it arrives, including completion/error handling. Buffering the entire response and then emitting one chunk removes the visible-latency benefit.
2. Do not mix HTTP connection semantics. Content framing and connection-close behavior must agree with the actual response to avoid SDK retries.
3. Check active server context capacity. A client configured for 8K can unnecessarily truncate requests to a 32K server, while a client assuming 32K can overflow a smaller slot. Reserve generation tokens and preserve instructions; reject an oversized request clearly rather than silently removing important history.
4. Keep serialized prompt prefixes stable when semantics permit. Changing timestamps, tool order, wrappers, or prefixes can reduce KV cache reuse. Do not delete important context just to obtain a cache hit.
5. Measure cold and repeat requests separately. A cached repeat is not a fresh-request speed measurement. UI first text, server TTFT, total completion time, and tokens/s measure different things.
6. Handle disconnects and cancellation so the server slot is released. Tool/structured-output validation may require different buffering than ordinary text.

These are implementation notes, not an installed patch. Client and llama.cpp APIs change; verify your versions before adapting a provider.
