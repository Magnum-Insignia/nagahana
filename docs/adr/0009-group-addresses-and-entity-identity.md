# ADR-0009: Group addresses, and one entity per machine inferred from traffic

- **Status:** Accepted (owner, 2026-10-01)
- **Decision IDs:** D-47, D-48; related: D-39, D-41, D-32
- **Sources:** owner, 2026-10-01, chat; `nagahana-app/docs/contract-topology-v2.md` §1 (the data
  contract between this repository and the app); RFC 826 (ARP); RFC 919 and RFC 922 (broadcasting,
  and broadcasting in the presence of subnets); RFC 5771 (IPv4 multicast, 224.0.0.0/4); RFC 4291
  §2.5.6 (IPv6 link-local addresses, fe80::/10) and §2.7 (IPv6 multicast, ff00::/8); RFC 4795 §2
  (LLMNR: 224.0.0.252 and ff02::1:3, port 5355); RFC 1002 (NetBIOS name and datagram services, UDP
  137 and 138); RFC 4861 (IPv6 Neighbor Discovery, hop limit 255); MITRE ATT&CK T1557.001
  (LLMNR/NBT-NS Poisoning and SMB Relay); Linux packet(7) (`sll_pkttype`)

## Context
The PCAP adapter (version 1.0.1) mapped addresses to entities by their value alone:
- every multicast address, and every IPv4 address ending in .255, became a `subnet`;
- every other address became a `host` or an `external`, one entity per address.

It decoded each frame's Ethernet addresses and discarded them. On the 30-minute CIC-IDS2018 sample
this gave 4 "subnets" that are multicast groups (224.0.0.22, 224.0.0.252, ff02::16, ff02::1:3) and
counted 5 IPv6 link-local addresses as extra machines. The DNS server and the router were plain
hosts.

The .255 rule is wrong in general. Which address is a subnet's broadcast depends on the prefix
length (RFC 922), and a capture does not carry prefix lengths. In a /20 such as 172.31.64.0/20,
172.31.65.255 is an ordinary host address.

Group addresses matter for security. LLMNR/NBT-NS poisoning (ATT&CK T1557.001) answers name queries
that were sent to them:
- LLMNR queries go to 224.0.0.252 and ff02::1:3 (RFC 4795 §2);
- NBT-NS queries go to the subnet broadcast (RFC 1002).

## Decision
1. **Group addresses (D-47).** A destination that names a group of machines is an entity of its own
   kind, `multicast`:
   - IPv4 224.0.0.0/4;
   - IPv6 ff00::/8;
   - IPv4 255.255.255.255;
   - a directed broadcast, i.e. an IPv4 destination sent to the Ethernet broadcast address that has
     not been seen sending.

   If such an address later sends, it is classified by the normal rule from then on. The kind
   `subnet` stays for real subnets.
2. **Entity identity (D-48).** One entity per machine:
   - a machine's IPv6 link-local addresses are merged into it as aliases, matched through the
     Ethernet address that ARP shows for its IPv4 address;
   - roles and names are inferred from the traffic itself, never from hard-coded addresses;
   - every inferred fact is time-stamped with the event time from which it was known (`since`).

## Consequences
- `datamodel/records.py`: `ENTITY_KINDS` gains `multicast`.
- `lab/sizing.py`: the L budget counts 8 node kinds instead of 7. CVG-AE has per-kind
  weights, so the L total grows from 932.8 M to 939.1 M parameters
  (the pre-build budget of that date; the built model has 1,134,268,667, `docs/sizing.md`).
- `ingest/pcap.py`, adapter 1.1.0. The module docstring states each rule precisely.
  - It keeps each frame's link addresses. A Linux cooked capture gives only the source, plus a packet
    type that marks a link broadcast (packet(7)). A raw-IP capture gives none, and the link-layer
    rules do not apply.
  - It learns MAC bindings from ARP only (RFC 826).
  - It aliases an IPv6 link-local source to the single host whose ARP-bound MAC it used. Flows that
    start from then on resolve to the host; earlier flows keep the separate entity.
  - It infers the roles `dns-server` and `gateway`, and names from DNS answers and TLS server names.
  - It adds the entity columns `first_sent`, `mac`, `ttl_initial` and `ttl_since`.
  - `columnar()` gains the tables `aliases`, `roles` and `names` (`datamodel/columnar.py`,
    `FACT_TABLES`).
  - `updates()` and `columnar()` still agree cell by cell. On the sample, the value and status
    matrices are bit-identical to those of 1.0.1.
- **Never beyond now.** The fact tables are parsed from the whole file, so they hold facts that are
  in the future of any earlier time. A consumer showing the state at time t keeps only rows with
  `since <= t`. The `mac` entity column has no time of its own.
- **Evidence on the sample** (adapter 1.1.0; each value was checked by an independent computation):
  - 5 multicast entities and no `subnet`;
  - 172.31.69.31 is a directed broadcast (NetBIOS datagrams, UDP 138, from 5 hosts; it never sends);
  - 25 ARP bindings, one MAC per IPv4 address;
  - 5 aliases, each made at the first frame of its link-local address, so no separate entity remains;
  - `gateway` on 172.31.69.1, whose MAC carries all external traffic;
  - `dns-server` on 172.31.0.2.

  172.31.64.1 appears only as the target of ARP requests (1,584 of them, none answered in the
  capture) and never in an IP packet, so no rule gives it a role.
- **Known limits:**
  - Proxy ARP binds one MAC to several IPv4 addresses, and a MAC bound to more than one address
    gives no alias.
  - An IPv4 address bound to several MACs (for example under ARP spoofing, T1557.002) keeps its
    first MAC in the `mac` column.
  - `ttl_initial` takes the largest TTL or hop limit an entity sent. IPv6 Neighbor Discovery always
    uses hop limit 255 (RFC 4861), so it raises an aliased host's value to 255. This is to be
    reviewed if `ttl_initial` is used to guess the operating system.
- **Not decided:**
  - merging other addresses of one machine (several IPv4 addresses, global IPv6 addresses,
    addresses that change hands such as DHCP leases);
  - identity for sources without link-layer addresses (raw-IP captures, flow records).
