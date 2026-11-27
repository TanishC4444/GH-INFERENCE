"""Offline VLM transport and registration checks; no weights are downloaded."""
import time
import io
import os
import sys
import tempfile
import hashlib
import types
import unittest
from unittest.mock import Mock, patch

from gh_inference import Agent, MODELS, add_vision_model, GitHubAIRunner
from gh_ai_runner.integrity import request_sha256
from gh_ai_runner.limits import Limits
from gh_ai_runner.worker_agent import AGENT_SCRIPT
from gh_ai_runner.runner import INFERENCE_SCRIPT, WORKFLOW_YAML


class VisionTests(unittest.TestCase):
    def test_registration_pins_projector_and_accounts_for_size(self):
        cfg = dict(MODELS["qwen0.5b"])
        cfg["files"] = [{"filename": "weights.gguf", "url": "https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/weights.gguf"}]
        with patch.dict(MODELS, {"test-vision-base": cfg}), patch(
            "gh_ai_runner.hub.resolve_model", return_value="test-vision-base"
        ), patch("gh_ai_runner.hub._info", return_value={"siblings": [{
            "rfilename": "mmproj.gguf", "size": 100,
            "lfs": {"sha256": "b" * 64}}]}):
            key = add_vision_model("owner/repo", filename="weights.gguf", projector_filename="mmproj.gguf")
            try:
                self.assertEqual(MODELS[key]["size_bytes"], cfg["size_bytes"] + 100)
                self.assertIn("a" * 40, MODELS[key]["files"][-1]["url"])
                self.assertNotIn("vision_chat_format", MODELS["test-vision-base"])
            finally:
                MODELS.pop(key)

    def test_text_model_rejects_images_before_dispatch(self):
        runner = Mock()
        with self.assertRaisesRegex(ValueError, "add_vision_model"):
            Agent(runner=runner).run("describe", images=["https://example.com/image.png"])
        runner.submit.assert_not_called()

    def test_worker_sends_multimodal_content(self):
        llm = Mock()
        llm.tokenize.return_value = [1]
        llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "red square"}, "finish_reason": "stop"}], "usage": {}}
        ns = {"time": time, "os": Mock(), "DEFAULT_LIMITS": Limits().to_dict(),
              "model_config": lambda _: {"n_ctx": 4096, "vision_chat_format": "moondream2"},
              "get_local_model": lambda *_: llm, "_LOADED_MODELS": {},
              "_MODEL_STATS": {("vision", 4096): {"download_seconds": 0, "load_seconds": 0, "weights_cached": False}}}
        ns["os"].environ.get.return_value = "vision"
        exec(AGENT_SCRIPT, ns)
        messages = [{"role": "system", "content": "help"}, {"role": "user", "content": "describe"}]
        result = ns["run_agent"]({"images": ["https://example.com/image.png"], "remember": False}, messages, {})
        self.assertEqual(result["results"][0]["output"], "red square")
        content = llm.create_chat_completion.call_args.kwargs["messages"][-1]["content"]
        self.assertEqual(content[1]["image_url"]["url"], "https://example.com/image.png")
        self.assertEqual(messages[-1]["content"], "describe")

    def worker(self, cfg=None):
        llm = Mock()
        llm.tokenize.return_value = [1]
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
        ns = {"time": time, "os": types.SimpleNamespace(environ={"LOCAL_MODEL": "vision"}),
              "DEFAULT_LIMITS": Limits().to_dict(),
              "model_config": lambda _: cfg or {"n_ctx": 2048, "vision_chat_format": "moondream2"},
              "get_local_model": Mock(return_value=llm), "_LOADED_MODELS": {},
              "_MODEL_STATS": {("vision", 2048): {
                  "download_seconds": 0, "load_seconds": 0, "weights_cached": False}}}
        exec(AGENT_SCRIPT, ns)
        return ns, llm

    def messages(self):
        return [{"role": "system", "content": "help"}, {"role": "user", "content": "describe"}]

    def test_invalid_images_rejected_on_client_without_dispatch(self):
        runner = Mock()
        agent = Agent(runner=runner)
        for images in ([], ["https://a", "https://b"], "https://a", [None],
                       ["file:///tmp/a"], ["http://a"], ["https://user:pw@a"],
                       ["https://a/#fragment"], ["https://a/white space"]):
            with self.subTest(images=images), self.assertRaises(ValueError):
                agent.run("describe", images=images)
        runner.submit.assert_not_called()

    def test_worker_rejects_invalid_images_before_model_load(self):
        for images in ([], "https://a", [None], ["http://a"], ["https://a", "https://b"],
                       ["https://user:pw@a"], ["https://a/#fragment"], ["https://a/white space"]):
            ns, _ = self.worker()
            with self.subTest(images=images), self.assertRaises(ValueError):
                ns["run_agent"]({"images": images}, self.messages(), {})
            ns["get_local_model"].assert_not_called()

    def test_worker_rejects_image_refinement_and_template_override(self):
        for agent, config in (({"refine_prompt": True}, {}),
                              ({}, {"model_options": {"chat_template_options": {"enable_thinking": False}}})):
            ns, _ = self.worker()
            with self.assertRaises(ValueError):
                ns["run_agent"](dict(agent, images=["https://example.com/a.png"]), self.messages(), config)
            ns["get_local_model"].assert_not_called()

    def test_image_fits_2048_context_and_usage_is_preserved(self):
        ns, llm = self.worker()
        result = ns["run_agent"]({"images": ["https://example.com/a.png"]}, self.messages(), {})
        self.assertEqual(result["usage"]["total_tokens"], 5)
        self.assertEqual(result["history"][0]["content"], "describe")
        self.assertEqual(llm.create_chat_completion.call_count, 1)

    def test_text_requests_still_send_plain_content(self):
        ns, llm = self.worker()
        result = ns["run_agent"]({}, self.messages(), {})
        self.assertEqual(result["results"][0]["output"], "answer")
        self.assertEqual(llm.create_chat_completion.call_args.kwargs["messages"][-1]["content"], "describe")

    def test_oversized_image_request_rejected_before_generation(self):
        ns, llm = self.worker()
        llm.tokenize.return_value = [1] * 1000
        with self.assertRaisesRegex(ValueError, "exceed context"):
            ns["run_agent"]({"images": ["https://example.com/a.png"]}, self.messages(), {})
        llm.create_chat_completion.assert_not_called()

    def test_vision_model_loads_matched_projector_once(self):
        ns = {}
        exec(INFERENCE_SCRIPT.split("\nstarted = time.monotonic()", 1)[0], ns)
        weights, projector = b"weights", b"projector"
        files = [{"filename": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                  "url": "https://example.com/" + name}
                 for name, data in (("weights.gguf", weights), ("mmproj.gguf", projector))]
        cfg = {"files": files, "size_gb": 1, "n_ctx": 2048, "max_n_ctx": 2048,
               "ram_per_1k_ctx_gb": .1, "vision_chat_format": "moondream2", "projector_filename": "mmproj.gguf"}
        ns["config"] = {"model_spec": cfg, "cache_weights": False}
        ns["memory_info"] = lambda: {"MemAvailable": int(32e9)}
        llama = types.ModuleType("llama_cpp")
        llama.Llama = Mock()
        handlers = types.ModuleType("llama_cpp.llama_chat_format")
        handlers.MoondreamChatHandler = Mock()
        handlers.Llava15ChatHandler = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "llama_cpp": llama, "llama_cpp.llama_chat_format": handlers}), patch(
            "urllib.request.urlopen", side_effect=[io.BytesIO(weights), io.BytesIO(projector)]
        ) as download:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                first = ns["get_local_model"]("vision", 2048)
                self.assertIs(first, ns["get_local_model"]("vision", 2048))
                self.assertEqual(download.call_count, 2)
                llama.Llama.assert_called_once()
                self.assertTrue(llama.Llama.call_args.kwargs["logits_all"])
                self.assertEqual(llama.Llama.call_args.kwargs["chat_handler"], handlers.MoondreamChatHandler.return_value)
                self.assertTrue(handlers.MoondreamChatHandler.call_args.kwargs["clip_model_path"].endswith("mmproj.gguf"))
            finally:
                os.chdir(previous)

    def test_workflow_transports_images_projector_and_request_integrity(self):
        import json
        cfg = dict(MODELS["qwen0.5b"])
        cfg.update(files=[{k: cfg[k] for k in ("filename", "url", "sha256", "size_bytes")}],
                   vision_chat_format="moondream2", projector_filename="mmproj.gguf")
        runner = GitHubAIRunner("offline-test-token", metadata_path=None, verbose=False)
        runner.prepare = Mock()
        runner._prepared = True
        runner._username = "tester"
        runner._default_branch = "main"
        runner._repo_id = 1
        with patch.dict(MODELS, {"test-vision": cfg}), patch("gh_ai_runner.client.requests.post") as post:
            post.return_value.status_code = 204
            job = runner.submit("describe", model="test-vision", n_ctx=2048,
                                agent_config={"images": ["https://example.com/a.png"]})
            config = json.loads(runner._workflow_inputs(job)["config"])
            self.assertEqual(config["agent"]["images"], ["https://example.com/a.png"])
            self.assertEqual(config["model_spec"]["projector_filename"], "mmproj.gguf")
            before = request_sha256(job.request)
            changed = dict(job.request, agent={"images": ["https://example.com/b.png"]})
            self.assertNotEqual(before, request_sha256(changed))

    def test_registration_rejects_bad_projector_metadata_and_path(self):
        cfg = dict(MODELS["qwen0.5b"])
        cfg["files"] = [{"filename": "weights.gguf", "url": "https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/weights.gguf"}]
        with patch.dict(MODELS, {"test-base": cfg}), patch(
            "gh_ai_runner.hub.resolve_model", return_value="test-base"
        ), patch("gh_ai_runner.hub._info", return_value={"siblings": []}):
            for name in ("../mmproj.gguf", "/mmproj.gguf", "a/../b.gguf", "missing.gguf"):
                with self.subTest(filename=name), self.assertRaises(ValueError):
                    add_vision_model("owner/repo", filename="weights.gguf", projector_filename=name)
        with self.assertRaises(ValueError):
            add_vision_model("owner/repo", filename="weights.gguf", projector_filename="mmproj.gguf", chat_format="unsupported")

    def test_corrupt_projector_download_is_rejected_and_partial_removed(self):
        ns = {}
        exec(INFERENCE_SCRIPT.split("\nstarted = time.monotonic()", 1)[0], ns)
        cfg = {"files": [{"filename": "weights.gguf", "size_bytes": 7,
                         "sha256": hashlib.sha256(b"weights").hexdigest(), "url": "https://example.com/w"},
                        {"filename": "mmproj.gguf", "size_bytes": 9,
                         "sha256": hashlib.sha256(b"projector").hexdigest(), "url": "https://example.com/p"}],
               "size_gb": 1, "n_ctx": 2048, "max_n_ctx": 2048, "ram_per_1k_ctx_gb": .1}
        ns["config"] = {"model_spec": cfg, "cache_weights": False}
        ns["memory_info"] = lambda: {"MemAvailable": int(32e9)}
        llama = types.ModuleType("llama_cpp")
        llama.Llama = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"llama_cpp": llama}), patch(
            "urllib.request.urlopen", side_effect=[io.BytesIO(b"weights"), io.BytesIO(b"corrupted")]):
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    ns["get_local_model"]("vision", 2048)
                self.assertFalse(os.path.exists("model_cache/vision/mmproj.gguf"))
                self.assertFalse(os.path.exists("model_cache/vision/mmproj.gguf.part"))
                llama.Llama.assert_not_called()
            finally:
                os.chdir(previous)

    def test_embedded_worker_compiles(self):
        compile(INFERENCE_SCRIPT, "worker", "exec")
        self.assertIn("Pillow", WORKFLOW_YAML)


if __name__ == "__main__":
    unittest.main()
