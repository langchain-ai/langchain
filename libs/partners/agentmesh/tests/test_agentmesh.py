"""Tests for AgentMesh LangChain integration."""

import pytest
from langchain_agentmesh import (
    CMVKIdentity,
    CMVKSignature,
    TrustedAgentCard,
    TrustHandshake,
    TrustPolicy,
    TrustGatedTool,
    TrustedToolExecutor,
    TrustCallbackHandler,
    DelegationChain,
)


class TestCMVKIdentity:
    """Tests for CMVKIdentity class."""

    def test_generate_identity(self):
        """Test identity generation."""
        identity = CMVKIdentity.generate(
            agent_name="test-agent",
            capabilities=["read", "write"]
        )
        
        assert identity.did.startswith("did:cmvk:")
        assert identity.agent_name == "test-agent"
        assert identity.public_key
        assert identity.private_key
        assert identity.capabilities == ["read", "write"]

    def test_sign_and_verify(self):
        """Test signing and verification."""
        identity = CMVKIdentity.generate("signer-agent")
        data = "test data to sign"
        
        signature = identity.sign(data)
        
        assert signature.public_key == identity.public_key
        assert signature.signature
        assert identity.verify_signature(data, signature)

    def test_verify_fails_wrong_data(self):
        """Test verification fails with wrong data."""
        identity = CMVKIdentity.generate("signer-agent")
        signature = identity.sign("original data")
        
        # Verification should fail with different data
        assert not identity.verify_signature("tampered data", signature)

    def test_public_identity(self):
        """Test public identity excludes private key."""
        identity = CMVKIdentity.generate("test-agent")
        public = identity.public_identity()
        
        assert public.did == identity.did
        assert public.public_key == identity.public_key
        assert public.private_key is None


class TestTrustedAgentCard:
    """Tests for TrustedAgentCard class."""

    def test_create_and_sign_card(self):
        """Test card creation and signing."""
        identity = CMVKIdentity.generate("card-agent", ["capability1"])
        
        card = TrustedAgentCard(
            name="Test Agent",
            description="A test agent",
            capabilities=["capability1", "capability2"],
        )
        card.sign(identity)
        
        assert card.identity is not None
        assert card.card_signature is not None
        assert card.verify_signature()

    def test_serialization(self):
        """Test card JSON serialization."""
        identity = CMVKIdentity.generate("json-agent")
        card = TrustedAgentCard(
            name="JSON Agent",
            description="Tests JSON",
            capabilities=["serialize"],
        )
        card.sign(identity)
        
        json_data = card.to_json()
        restored = TrustedAgentCard.from_json(json_data)
        
        assert restored.name == card.name
        assert restored.capabilities == card.capabilities
        assert restored.identity.did == card.identity.did


class TestTrustHandshake:
    """Tests for TrustHandshake class."""

    def test_verify_valid_peer(self):
        """Test verification of a valid peer."""
        my_identity = CMVKIdentity.generate("my-agent")
        peer_identity = CMVKIdentity.generate("peer-agent", ["required_cap"])
        
        peer_card = TrustedAgentCard(
            name="Peer Agent",
            description="A peer",
            capabilities=["required_cap"],
        )
        peer_card.sign(peer_identity)
        
        handshake = TrustHandshake(my_identity)
        result = handshake.verify_peer(
            peer_card,
            required_capabilities=["required_cap"]
        )
        
        assert result.trusted
        assert result.trust_score == 1.0

    def test_verify_missing_capability(self):
        """Test verification fails for missing capability."""
        my_identity = CMVKIdentity.generate("my-agent")
        peer_identity = CMVKIdentity.generate("peer-agent", ["cap1"])
        
        peer_card = TrustedAgentCard(
            name="Peer Agent",
            description="A peer",
            capabilities=["cap1"],
        )
        peer_card.sign(peer_identity)
        
        handshake = TrustHandshake(my_identity)
        result = handshake.verify_peer(
            peer_card,
            required_capabilities=["cap1", "cap2"]
        )
        
        assert not result.trusted
        assert "Missing required capabilities" in result.reason

    def test_cache_ttl(self):
        """Test that verification results are cached."""
        my_identity = CMVKIdentity.generate("my-agent")
        peer_identity = CMVKIdentity.generate("peer-agent")
        
        peer_card = TrustedAgentCard(
            name="Peer Agent",
            description="A peer",
            capabilities=[],
        )
        peer_card.sign(peer_identity)
        
        handshake = TrustHandshake(my_identity)
        
        # First verification
        result1 = handshake.verify_peer(peer_card)
        # Second should use cache
        result2 = handshake.verify_peer(peer_card)
        
        assert result1.trusted == result2.trusted


class TestDelegationChain:
    """Tests for DelegationChain class."""

    def test_add_delegation(self):
        """Test adding a delegation."""
        root = CMVKIdentity.generate("root-agent")
        worker_identity = CMVKIdentity.generate("worker-agent")
        
        worker_card = TrustedAgentCard(
            name="Worker",
            description="Worker agent",
            capabilities=[],
        )
        worker_card.sign(worker_identity)
        
        chain = DelegationChain(root)
        delegation = chain.add_delegation(
            delegatee=worker_card,
            capabilities=["read", "write"],
            expires_in_hours=24,
        )
        
        assert delegation.delegator == root.did
        assert delegation.delegatee == worker_identity.did
        assert "read" in delegation.capabilities

    def test_verify_chain(self):
        """Test chain verification."""
        root = CMVKIdentity.generate("root-agent")
        worker_identity = CMVKIdentity.generate("worker-agent")
        
        worker_card = TrustedAgentCard(
            name="Worker",
            description="Worker agent",
            capabilities=[],
        )
        worker_card.sign(worker_identity)
        
        chain = DelegationChain(root)
        chain.add_delegation(
            delegatee=worker_card,
            capabilities=["read"],
        )
        
        assert chain.verify()


class TestTrustGatedTool:
    """Tests for TrustGatedTool class."""

    def test_can_invoke_with_capability(self):
        """Test capability check for tool invocation."""
        my_identity = CMVKIdentity.generate("executor")
        invoker_identity = CMVKIdentity.generate("invoker", ["database"])
        
        def mock_tool(query: str) -> str:
            return f"Result: {query}"
        
        gated_tool = TrustGatedTool(
            tool=mock_tool,
            required_capabilities=["database"],
        )
        
        invoker_card = TrustedAgentCard(
            name="Invoker",
            description="Has database cap",
            capabilities=["database"],
        )
        invoker_card.sign(invoker_identity)
        
        handshake = TrustHandshake(my_identity)
        result = gated_tool.can_invoke(invoker_card, handshake)
        
        assert result.trusted


class TestTrustCallbackHandler:
    """Tests for TrustCallbackHandler class."""

    def test_event_logging(self):
        """Test that events are logged."""
        identity = CMVKIdentity.generate("callback-agent")
        policy = TrustPolicy(audit_all_calls=True)
        
        handler = TrustCallbackHandler(identity, policy)
        
        # Simulate some events
        from uuid import uuid4
        run_id = uuid4()
        
        handler.on_llm_start(
            {"name": "test-model"},
            ["prompt"],
            run_id=run_id,
        )
        
        events = handler.get_events()
        assert len(events) == 1
        assert events[0].event_type == "llm_start"

    def test_trust_summary(self):
        """Test trust summary generation."""
        identity = CMVKIdentity.generate("summary-agent")
        handler = TrustCallbackHandler(identity)
        
        summary = handler.get_trust_summary()
        
        assert "total_events" in summary
        assert "verified_events" in summary
        assert "verification_rate" in summary
