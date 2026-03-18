# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding/reranker engine keepalive and mx.compile integration."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


class TestTryCompile:
    """Tests for _try_compile() primitive-output compile path.

    After c9d67d6 the _CompiledForward wrapper class was replaced by an
    inline _compiled_embed closure stored on self._compiled_embed.  The
    raw self.model is never wrapped.
    """

    def test_try_compile_success(self):
        """Should return True and populate _compiled_embed."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        mock_raw_model = MagicMock()
        model.model = mock_raw_model

        mock_compiled_fn = MagicMock(return_value=MagicMock())

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.return_value = mock_compiled_fn
            mock_mx.zeros.return_value = MagicMock()
            mock_mx.int32 = "int32"
            result = model._try_compile()

        assert result is True
        assert model._compiled_embed is mock_compiled_fn
        # model.model must remain the original unwrapped model
        assert model.model is mock_raw_model

    def test_try_compile_failure_leaves_model_unchanged(self):
        """Should revert to uncompiled state on compilation failure."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        original_model = MagicMock()
        model.model = original_model

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.side_effect = RuntimeError("compile failed")
            result = model._try_compile()

        assert result is False
        assert model._is_compiled is False
        assert model._compiled_embed is None
        assert model.model is original_model

    def test_try_compile_warmup_failure_reverts(self):
        """If compile succeeds but warmup forward pass fails, revert."""
        from omlx.models.embedding import MLXEmbeddingModel

        model = MLXEmbeddingModel("test-model")
        model.model = MagicMock()

        mock_compiled_fn = MagicMock(side_effect=RuntimeError("eval error"))

        with patch("omlx.models.embedding.mx") as mock_mx:
            mock_mx.compile.return_value = mock_compiled_fn
            mock_mx.zeros.return_value = MagicMock()
            mock_mx.int32 = "int32"
            result = model._try_compile()

        assert result is False
        assert model._compiled_embed is None


class TestEmbeddingEngineKeepalive:
    """Tests for keepalive integration in EmbeddingEngine."""

    def test_compiled_model_no_keepalive(self):
        """Keepalive should NOT start if mx.compile succeeded."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = True
            mock_model.hidden_size = 384
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

        assert engine._keepalive_task is None

    def test_uncompiled_model_starts_keepalive(self):
        """Keepalive SHOULD start if mx.compile failed."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            mock_model.hidden_size = 384
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()

            asyncio.run(_run())

    def test_keepalive_stops_on_engine_stop(self):
        """Keepalive task should be cancelled on engine stop."""
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()
                assert engine._keepalive_task is None

            asyncio.run(_run())

    def test_active_requests_tracking(self):
        """_active_requests should be incremented during embed calls."""
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.models.embedding import EmbeddingOutput

        engine = EmbeddingEngine("test-model")

        with patch("omlx.engine.embedding.MLXEmbeddingModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = True
            mock_model.embed.return_value = EmbeddingOutput(
                embeddings=[[0.1]], total_tokens=1, dimensions=1,
            )
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._active_requests == 0
                await engine.embed(["test"])
                assert engine._active_requests == 0  # Back to 0 after completion

            asyncio.run(_run())


class TestRerankerEngineKeepalive:
    """Tests for keepalive integration in RerankerEngine."""

    def test_compiled_model_no_keepalive(self):
        """Keepalive should NOT start if mx.compile succeeded."""
        from omlx.engine.reranker import RerankerEngine

        engine = RerankerEngine("test-model")

        with patch("omlx.engine.reranker.MLXRerankerModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = True
            MockModel.return_value = mock_model

            asyncio.run(engine.start())

        assert engine._keepalive_task is None

    def test_uncompiled_model_starts_keepalive(self):
        """Keepalive SHOULD start if mx.compile failed."""
        from omlx.engine.reranker import RerankerEngine

        engine = RerankerEngine("test-model")

        with patch("omlx.engine.reranker.MLXRerankerModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()

            asyncio.run(_run())

    def test_keepalive_stops_on_engine_stop(self):
        """Keepalive task should be cancelled on engine stop."""
        from omlx.engine.reranker import RerankerEngine

        engine = RerankerEngine("test-model")

        with patch("omlx.engine.reranker.MLXRerankerModel") as MockModel:
            mock_model = MagicMock()
            mock_model._is_compiled = False
            MockModel.return_value = mock_model

            async def _run():
                await engine.start()
                assert engine._keepalive_task is not None
                await engine.stop()
                assert engine._keepalive_task is None

            asyncio.run(_run())
