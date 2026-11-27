MODELS = {
    "tinyllama": {
        "name":      "TinyLlama 1.1B Chat",
        "size_gb":   0.6,
        "n_ctx":     2048,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename":  "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    },
    "llama": {
        "name":      "Llama 3.2 1B Instruct",
        "size_gb":   0.7,
        "n_ctx":     4096,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename":  "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },

    # ── ADD NEW MODELS BELOW THIS LINE ──────────────────────────────────────
    # Rules:
    #   size_gb + (extra_ctx/1000)*0.2 + 1.0  must stay under RUNNER_RAM_GB (7)
    #   Always use Q4_K_M GGUF quantisation for the best size/quality tradeoff
    #   After adding, also add the filename to INFERENCE_SCRIPT in runner.py
    # ────────────────────────────────────────────────────────────────────────

    "phi3": {
        "name":      "Phi-3.5 Mini Instruct",
        "size_gb":   2.2,
        "n_ctx":     4096,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "filename":  "Phi-3.5-mini-instruct-Q4_K_M.gguf",
    },
    "qwen": {
        "name":      "Qwen 2.5 1.5B Instruct",
        "size_gb":   1.0,
        "n_ctx":     4096,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "filename":  "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
    "gemma2": {
        "name":      "Gemma 2 2B Instruct",
        "size_gb":   1.6,
        "n_ctx":     4096,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "filename":  "gemma-2-2b-it-Q4_K_M.gguf",
    },
    "deepseek": {
        "name":      "DeepSeek-R1 1.5B",
        "size_gb":   1.1,
        "n_ctx":     4096,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "filename":  "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    },
}

RUNNER_RAM_GB     = 7
RAM_PER_1K_CTX_GB = 0.2
MAX_TOKENS_LIMIT  = 4096
MAX_TEMPERATURE   = 2.0
MIN_TEMPERATURE   = 0.0