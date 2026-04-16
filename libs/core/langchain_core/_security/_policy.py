"""SSRF protection policy with IP validation and DNS-aware URL checking."""

import asyncio
import dataclasses
import ipaddress
import os
import socket
import urllib.parse

from langchain_core._security._exceptions import SSRFBlockedError

# ---------------------------------------------------------------------------
# Blocklist constants
# ---------------------------------------------------------------------------

_BLOCKED_IPV4_NETWORKS: tuple[ipaddress.IPv4Network, ...] = tuple(
    ipaddress.IPv4Network(n)
    for n in (
        "10.0.0.0/8",  # RFC 1918 - private class A
        "172.16.0.0/12",  # RFC 1918 - private class B
        "192.168.0.0/16",  # RFC 1918 - private class C
        "127.0.0.0/8",  # RFC 1122 - loopback
        "169.254.0.0/16",  # RFC 3927 - link-local
        "0.0.0.0/8",  # RFC 1122 - "this network"
        "100.64.0.0/10",  # RFC 6598 - shared/CGN address space
        "192.0.0.0/24",  # RFC 6890 - IETF protocol assignments
        "192.0.2.0/24",  # RFC 5737 - TEST-NET-1 (documentation)
        "198.18.0.0/15",  # RFC 2544 - benchmarking
        "198.51.100.0/24",  # RFC 5737 - TEST-NET-2 (documentation)
        "203.0.113.0/24",  # RFC 5737 - TEST-NET-3 (documentation)
        "224.0.0.0/4",  # RFC 5771 - multicast
        "240.0.0.0/4",  # RFC 1112 - reserved for future use
        "255.255.255.255/32",  # RFC 919  - limited broadcast
    )
)

_BLOCKED_IPV6_NETWORKS: tuple[ipaddress.IPv6Network, ...] = tuple(
    ipaddress.IPv6Network(n)
    for n in (
        "::1/128",  # RFC 4291 - loopback
        "fc00::/7",  # RFC 4193 - unique local addresses (ULA)
        "fe80::/10",  # RFC 4291 - link-local
        "ff00::/8",  # RFC 4291 - multicast
        "::ffff:0:0/96",  # RFC 4291 - IPv4-mapped IPv6 addresses
        "::0.0.0.0/96",  # RFC 4291 - IPv4-compatible IPv6 (deprecated)
        "64:ff9b::/96",  # RFC 6052 - NAT64 well-known prefix
        "64:ff9b:1::/48",  # RFC 8215 - NAT64 discovery prefix
    )
)

_CLOUD_METADATA_IPS: frozenset[str] = frozenset(
    {
        "169.254.169.254",  # AWS, GCP, Azure, DigitalOcean, Oracle Cloud
        "169.254.170.2",  # AWS ECS task metadata
        "169.254.170.23",  # AWS EKS Pod Identity Agent
        "100.100.100.200",  # Alibaba Cloud metadata
        "fd00:ec2::254",  # AWS EC2 IMDSv2 over IPv6 (Nitro instances)
        "fd00:ec2::23",  # AWS EKS Pod Identity Agent (IPv6)
        "fe80::a9fe:a9fe",  # OpenStack Nova metadata (IPv6 link-local)
    }
)

# Network ranges that are always blocked when block_cloud_metadata=True,
# independent of block_private_ips.  The entire link-local range is used by
# cloud metadata services across providers.
_CLOUD_METADATA_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.IPv4Network("169.254.0.0/16"),
)

_CLOUD_METADATA_HOSTNAMES: frozenset[str] = frozenset(
    {
        "metadata.google.internal",
        "metadata.amazonaws.com",
        "metadata",
        "instance-data",
    }
)

_LOCALHOST_NAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "host.docker.internal",
    }
)

_K8S_SUFFIX = ".svc.cluster.local"

_LOOPBACK_IPV4 = ipaddress.IPv4Network("127.0.0.0/8")
_LOOPBACK_IPV6 = ipaddress.IPv6Address("::1")

# NAT64 well-known prefixes
_NAT64_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")
_NAT64_DISCOVERY_PREFIX = ipaddress.IPv6Network("64:ff9b:1::/48")


# ---------------------------------------------------------------------------
# SSRFPolicy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SSRFPolicy:
    """Immutable policy controlling which URLs/IPs are considered safe."""

    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    block_private_ips: bool = True
    block_localhost: bool = True
    block_cloud_metadata: bool = True
    block_k8s_internal: bool = True
    allowed_hosts: frozenset[str] = frozenset()
    additional_blocked_cidrs: tuple[
        ipaddress.IPv4Network | ipaddress.IPv6Network, ...
    ] = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_embedded_ipv4(
    addr: ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | None:
    """Extract an embedded IPv4 from IPv4-mapped or NAT64 IPv6 addresses."""
    # Check ipv4_mapped first (covers ::ffff:x.x.x.x)
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped

    # Check NAT64 prefixes — embedded IPv4 is in the last 4 bytes
    if addr in _NAT64_PREFIX or addr in _NAT64_DISCOVERY_PREFIX:
        raw = addr.packed
        return ipaddress.IPv4Address(raw[-4:])

    return None


def _ip_in_blocked_networks(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
    policy: SSRFPolicy,
) -> str | None:
    """Return a reason string if *addr* falls in a blocked range, else None."""
    # NOTE: if profiling shows this is a hot path, consider memoising with
    # @functools.lru_cache (key on (addr, id(policy))).
    if isinstance(addr, ipaddress.IPv4Address):
        if policy.block_private_ips:
            for net in _BLOCKED_IPV4_NETWORKS:
                if addr in net:
                    return "private IP range"
        for net in policy.additional_blocked_cidrs:  # type: ignore[assignment]
            if isinstance(net, ipaddress.IPv4Network) and addr in net:
                return "blocked CIDR"
    else:
        if policy.block_private_ips:
            for net in _BLOCKED_IPV6_NETWORKS:  # type: ignore[assignment]
                if addr in net:
                    return "private IP range"
        for net in policy.additional_blocked_cidrs:  # type: ignore[assignment]
            if isinstance(net, ipaddress.IPv6Network) and addr in net:
                return "blocked CIDR"

    # Loopback check — independent of block_private_ips so that
    # block_localhost=True still catches 127.x.x.x / ::1 even when
    # private IPs are allowed.
    if policy.block_localhost:
        if isinstance(addr, ipaddress.IPv4Address) and (
            addr in _LOOPBACK_IPV4 or addr in ipaddress.IPv4Network("0.0.0.0/8")
        ):
            return "localhost address"
        if isinstance(addr, ipaddress.IPv6Address) and addr == _LOOPBACK_IPV6:
            return "localhost address"

    # Cloud metadata check — IP set *and* network ranges (e.g. 169.254.0.0/16).
    # Independent of block_private_ips so that allow_private=True still blocks
    # cloud metadata endpoints.
    if policy.block_cloud_metadata:
        if str(addr) in _CLOUD_METADATA_IPS:
            return "cloud metadata endpoint"
        for net in _CLOUD_METADATA_NETWORKS:  # type: ignore[assignment]
            if addr in net:
                return "cloud metadata endpoint"

    return None


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------


def validate_resolved_ip(ip_str: str, policy: SSRFPolicy) -> None:
    """Validate a resolved IP address against the SSRF policy.

    Raises SSRFBlockedError if the IP is blocked.
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError as exc:
        raise SSRFBlockedError("invalid IP address") from exc

    if isinstance(addr, ipaddress.IPv6Address):
        inner = _extract_embedded_ipv4(addr)
        if inner is not None:
            addr = inner

    reason = _ip_in_blocked_networks(addr, policy)
    if reason is not None:
        raise SSRFBlockedError(reason)


def validate_hostname(hostname: str, policy: SSRFPolicy) -> None:
    """Validate a hostname against the SSRF policy.

    Raises SSRFBlockedError if the hostname is blocked.
    """
    lower = hostname.lower()

    if policy.block_localhost and lower in _LOCALHOST_NAMES:
        raise SSRFBlockedError("localhost address")

    if policy.block_cloud_metadata and lower in _CLOUD_METADATA_HOSTNAMES:
        raise SSRFBlockedError("cloud metadata endpoint")

    if policy.block_k8s_internal and lower.endswith(_K8S_SUFFIX):
        raise SSRFBlockedError("Kubernetes internal DNS")


def _effective_allowed_hosts(policy: SSRFPolicy) -> frozenset[str]:
    """Return allowed_hosts, augmented for local environments."""
    extra: set[str] = set()
    if os.environ.get("LANGCHAIN_ENV", "").startswith("local"):
        extra.update({"localhost", "testserver"})
    if extra:
        return policy.allowed_hosts | frozenset(extra)
    return policy.allowed_hosts


async def validate_url(url: str, policy: SSRFPolicy = SSRFPolicy()) -> None:
    """Validate a URL against the SSRF policy, including DNS resolution.

    This is the primary entry-point for async code paths. It delegates
    scheme/hostname/allowed-hosts checks to `validate_url_sync`, then
    resolves DNS and validates every resolved IP.

    Raises:
        SSRFBlockedError: If the URL violates the policy.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""

    validate_url_sync(url, policy)

    allowed = {h.lower() for h in _effective_allowed_hosts(policy)}
    if hostname.lower() in allowed:
        return

    scheme = (parsed.scheme or "").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        addrinfo = await asyncio.to_thread(
            socket.getaddrinfo, hostname, port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        msg = "DNS resolution failed"
        raise SSRFBlockedError(msg) from exc

    for _family, _type, _proto, _canonname, sockaddr in addrinfo:
        validate_resolved_ip(str(sockaddr[0]), policy)


def validate_url_sync(url: str, policy: SSRFPolicy = SSRFPolicy()) -> None:
    """Synchronous URL validation (no DNS resolution).

    Suitable for Pydantic validators and other sync contexts. Checks scheme
    and hostname patterns only - use `validate_url` for full DNS-aware checking.

    Raises:
        SSRFBlockedError: If the URL violates the policy.
    """
    parsed = urllib.parse.urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        msg = f"scheme '{scheme}' not allowed"
        raise SSRFBlockedError(msg)

    hostname = parsed.hostname
    if not hostname:
        msg = "missing hostname"
        raise SSRFBlockedError(msg)

    allowed = _effective_allowed_hosts(policy)
    if hostname.lower() in {h.lower() for h in allowed}:
        return

    try:
        ipaddress.ip_address(hostname)
        validate_resolved_ip(hostname, policy)
    except SSRFBlockedError:
        raise
    except ValueError:
        pass
    else:
        return

    validate_hostname(hostname, policy)
