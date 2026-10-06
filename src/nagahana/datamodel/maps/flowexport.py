"""Flow-export mapping tables: NetFlow v5, NetFlow v9 (RFC 3954), IPFIX (RFC 7011, RFC 5103 biflows,
RFC 6313 structured data) and sFlow v5.

`IPFIX_IES` transcribes the IANA "IP Flow Information Export (IPFIX) Entities" registry entries the
adapters decode (identifier, name, abstract data type, unit), following RFC 7012 for the abstract
types. NetFlow v9 field types 1 to 127 are the same elements (RFC 3954 section 8; RFC 7012 section 1),
so one table serves both. An element missing from the table, or an enterprise-specific element
without a definition, is retained as an uncatalogued attribute with its raw value (AS-695).

sFlow v5 layouts follow "sFlow Version 5" (sflow.org, July 2004): datagram header, flow samples
(format 1), counter samples (format 2), expanded flow samples (format 3), expanded counter samples
(format 4), flow records (raw packet header 1, Ethernet frame data 2, IPv4 data 3, IPv6 data 4,
extended switch 1001, extended router 1002, extended gateway 1003, extended user 1004, extended URL
1005) and counter records (generic interface 1, Ethernet interface 2, VLAN 5, processor 1001).
"""

from __future__ import annotations

from nagahana.datamodel.native import R, RecordMap, Row
from nagahana.datamodel.spec import Level

REF_V5 = "Cisco NetFlow Export Datagram Format, version 5"
REF_V9 = "RFC 3954 (NetFlow version 9)"
REF_IPFIX = "RFC 7011, RFC 7012, IANA IPFIX Information Elements registry, RFC 5103"
REF_SFLOW = "sFlow Version 5, sflow.org, July 2004"

#: Abstract data type (RFC 7011 section 6.1, RFC 6313) -> short dtype code of the data model.
ABSTRACT_DTYPES: dict[str, str] = {
    "octetArray": "y", "unsigned8": "i", "unsigned16": "i", "unsigned32": "i", "unsigned64": "i", "signed8": "i",
    "signed16": "i", "signed32": "i", "signed64": "i", "float32": "f", "float64": "f", "boolean": "b",
    "macAddress": "m", "string": "s", "dateTimeSeconds": "t", "dateTimeMilliseconds": "t",
    "dateTimeMicroseconds": "t", "dateTimeNanoseconds": "t", "ipv4Address": "a", "ipv6Address": "a",
    "basicList": "M", "subTemplateList": "M", "subTemplateMultiList": "M",
}

#: (element id, name, abstract data type, unit, shared target or "", converter). Targets marked "*" are
#: derived by the adapter (times, ICMP type/code split) and keep the raw element as a native field.
IPFIX_IES: tuple[tuple[int, str, str, str | None, str, str], ...] = (
    (1, "octetDeltaCount", "unsigned64", "octets", "*", ""),
    (2, "packetDeltaCount", "unsigned64", "packets", "*", ""),
    (3, "deltaFlowCount", "unsigned64", "flows", "", ""),
    (4, "protocolIdentifier", "unsigned8", None, "flow.protocol", "count"),
    (5, "ipClassOfService", "unsigned8", None, "flow.ip_tos", "count"),
    (6, "tcpControlBits", "unsigned16", None, "flow.tcp_flags_fwd", "tcp_bits8"),
    (7, "sourceTransportPort", "unsigned16", None, "flow.src_port", "port"),
    (8, "sourceIPv4Address", "ipv4Address", None, "flow.src_ip", "addr"),
    (9, "sourceIPv4PrefixLength", "unsigned8", "bits", "", ""),
    (10, "ingressInterface", "unsigned32", None, "flow.ingress_if", "count"),
    (11, "destinationTransportPort", "unsigned16", None, "flow.dst_port", "port"),
    (12, "destinationIPv4Address", "ipv4Address", None, "flow.dst_ip", "addr"),
    (13, "destinationIPv4PrefixLength", "unsigned8", "bits", "", ""),
    (14, "egressInterface", "unsigned32", None, "flow.egress_if", "count"),
    (15, "ipNextHopIPv4Address", "ipv4Address", None, "flow.next_hop", "addr"),
    (16, "bgpSourceAsNumber", "unsigned32", None, "flow.src_as", "count"),
    (17, "bgpDestinationAsNumber", "unsigned32", None, "flow.dst_as", "count"),
    (18, "bgpNextHopIPv4Address", "ipv4Address", None, "", ""),
    (19, "postMCastPacketDeltaCount", "unsigned64", "packets", "", ""),
    (20, "postMCastOctetDeltaCount", "unsigned64", "octets", "", ""),
    (21, "flowEndSysUpTime", "unsigned32", "ms", "*", ""),
    (22, "flowStartSysUpTime", "unsigned32", "ms", "*", ""),
    (23, "postOctetDeltaCount", "unsigned64", "octets", "", ""),
    (24, "postPacketDeltaCount", "unsigned64", "packets", "", ""),
    (25, "minimumIpTotalLength", "unsigned64", "octets", "pkt.ip_len_min", "count"),
    (26, "maximumIpTotalLength", "unsigned64", "octets", "pkt.ip_len_max", "count"),
    (27, "sourceIPv6Address", "ipv6Address", None, "flow.src_ip", "addr"),
    (28, "destinationIPv6Address", "ipv6Address", None, "flow.dst_ip", "addr"),
    (29, "sourceIPv6PrefixLength", "unsigned8", "bits", "", ""),
    (30, "destinationIPv6PrefixLength", "unsigned8", "bits", "", ""),
    (31, "flowLabelIPv6", "unsigned32", None, "", ""),
    (32, "icmpTypeCodeIPv4", "unsigned16", None, "*", ""),
    (33, "igmpType", "unsigned8", None, "", ""),
    (34, "samplingInterval", "unsigned32", "packets", "flow.sampling_rate", "count"),
    (35, "samplingAlgorithm", "unsigned8", None, "", ""),
    (36, "flowActiveTimeout", "unsigned16", "s", "", ""),
    (37, "flowIdleTimeout", "unsigned16", "s", "", ""),
    (38, "engineType", "unsigned8", None, "", ""),
    (39, "engineId", "unsigned8", None, "", ""),
    (40, "exportedOctetTotalCount", "unsigned64", "octets", "", ""),
    (41, "exportedMessageTotalCount", "unsigned64", "messages", "", ""),
    (42, "exportedFlowRecordTotalCount", "unsigned64", "flows", "", ""),
    (43, "ipv4RouterSc", "ipv4Address", None, "", ""),
    (44, "sourceIPv4Prefix", "ipv4Address", None, "", ""),
    (45, "destinationIPv4Prefix", "ipv4Address", None, "", ""),
    (46, "mplsTopLabelType", "unsigned8", None, "", ""),
    (47, "mplsTopLabelIPv4Address", "ipv4Address", None, "", ""),
    (48, "samplerId", "unsigned8", None, "", ""),
    (49, "samplerMode", "unsigned8", None, "", ""),
    (50, "samplerRandomInterval", "unsigned32", "packets", "flow.sampling_rate", "count"),
    (51, "classId", "unsigned8", None, "", ""),
    (52, "minimumTTL", "unsigned8", "hops", "pkt.ttl_min", "count"),
    (53, "maximumTTL", "unsigned8", "hops", "pkt.ttl_max", "count"),
    (54, "fragmentIdentification", "unsigned32", None, "", ""),
    (55, "postIpClassOfService", "unsigned8", None, "", ""),
    (56, "sourceMacAddress", "macAddress", None, "flow.src_mac", "mac"),
    (57, "postDestinationMacAddress", "macAddress", None, "", ""),
    (58, "vlanId", "unsigned16", None, "flow.vlan", "count"),
    (59, "postVlanId", "unsigned16", None, "", ""),
    (60, "ipVersion", "unsigned8", None, "flow.ip_version", "count"),
    (61, "flowDirection", "unsigned8", None, "", ""),
    (62, "ipNextHopIPv6Address", "ipv6Address", None, "flow.next_hop", "addr"),
    (63, "bgpNextHopIPv6Address", "ipv6Address", None, "", ""),
    (64, "ipv6ExtensionHeaders", "unsigned32", None, "", ""),
    (70, "mplsTopLabelStackSection", "octetArray", None, "", ""),
    (71, "mplsLabelStackSection2", "octetArray", None, "", ""),
    (72, "mplsLabelStackSection3", "octetArray", None, "", ""),
    (73, "mplsLabelStackSection4", "octetArray", None, "", ""),
    (74, "mplsLabelStackSection5", "octetArray", None, "", ""),
    (75, "mplsLabelStackSection6", "octetArray", None, "", ""),
    (76, "mplsLabelStackSection7", "octetArray", None, "", ""),
    (77, "mplsLabelStackSection8", "octetArray", None, "", ""),
    (78, "mplsLabelStackSection9", "octetArray", None, "", ""),
    (79, "mplsLabelStackSection10", "octetArray", None, "", ""),
    (80, "destinationMacAddress", "macAddress", None, "flow.dst_mac", "mac"),
    (81, "postSourceMacAddress", "macAddress", None, "", ""),
    (82, "interfaceName", "string", None, "", ""),
    (83, "interfaceDescription", "string", None, "", ""),
    (84, "samplerName", "string", None, "", ""),
    (85, "octetTotalCount", "unsigned64", "octets", "*", ""),
    (86, "packetTotalCount", "unsigned64", "packets", "*", ""),
    (87, "flagsAndSamplerId", "unsigned32", None, "", ""),
    (88, "fragmentOffset", "unsigned16", None, "", ""),
    (89, "forwardingStatus", "unsigned8", None, "flow.forwarding_status", "count"),
    (90, "mplsVpnRouteDistinguisher", "octetArray", None, "", ""),
    (91, "mplsTopLabelPrefixLength", "unsigned8", "bits", "", ""),
    (92, "srcTrafficIndex", "unsigned32", None, "", ""),
    (93, "dstTrafficIndex", "unsigned32", None, "", ""),
    (94, "applicationDescription", "string", None, "", ""),
    (95, "applicationId", "octetArray", None, "", ""),
    (96, "applicationName", "string", None, "flow.app_proto", "app_proto"),
    (98, "postIpDiffServCodePoint", "unsigned8", None, "", ""),
    (99, "multicastReplicationFactor", "unsigned32", None, "", ""),
    (100, "className", "string", None, "", ""),
    (101, "classificationEngineId", "unsigned8", None, "", ""),
    (102, "layer2packetSectionOffset", "unsigned16", "octets", "", ""),
    (103, "layer2packetSectionSize", "unsigned16", "octets", "", ""),
    (104, "layer2packetSectionData", "octetArray", None, "", ""),
    (128, "bgpNextAdjacentAsNumber", "unsigned32", None, "", ""),
    (129, "bgpPrevAdjacentAsNumber", "unsigned32", None, "", ""),
    (130, "exporterIPv4Address", "ipv4Address", None, "", ""),
    (131, "exporterIPv6Address", "ipv6Address", None, "", ""),
    (132, "droppedOctetDeltaCount", "unsigned64", "octets", "", ""),
    (133, "droppedPacketDeltaCount", "unsigned64", "packets", "", ""),
    (134, "droppedOctetTotalCount", "unsigned64", "octets", "", ""),
    (135, "droppedPacketTotalCount", "unsigned64", "packets", "", ""),
    (136, "flowEndReason", "unsigned8", None, "flow.end_reason", "ipfix_end_reason"),
    (137, "commonPropertiesId", "unsigned64", None, "", ""),
    (138, "observationPointId", "unsigned64", None, "", ""),
    (139, "icmpTypeCodeIPv6", "unsigned16", None, "*", ""),
    (140, "mplsTopLabelIPv6Address", "ipv6Address", None, "", ""),
    (141, "lineCardId", "unsigned32", None, "", ""),
    (142, "portId", "unsigned32", None, "", ""),
    (143, "meteringProcessId", "unsigned32", None, "", ""),
    (144, "exportingProcessId", "unsigned32", None, "", ""),
    (145, "templateId", "unsigned16", None, "", ""),
    (146, "wlanChannelId", "unsigned8", None, "", ""),
    (147, "wlanSSID", "string", None, "", ""),
    (148, "flowId", "unsigned64", None, "flow.uid", "int_text"),
    (149, "observationDomainId", "unsigned32", None, "", ""),
    (150, "flowStartSeconds", "dateTimeSeconds", "s", "*", ""),
    (151, "flowEndSeconds", "dateTimeSeconds", "s", "*", ""),
    (152, "flowStartMilliseconds", "dateTimeMilliseconds", "s", "*", ""),
    (153, "flowEndMilliseconds", "dateTimeMilliseconds", "s", "*", ""),
    (154, "flowStartMicroseconds", "dateTimeMicroseconds", "s", "*", ""),
    (155, "flowEndMicroseconds", "dateTimeMicroseconds", "s", "*", ""),
    (156, "flowStartNanoseconds", "dateTimeNanoseconds", "s", "*", ""),
    (157, "flowEndNanoseconds", "dateTimeNanoseconds", "s", "*", ""),
    (158, "flowStartDeltaMicroseconds", "unsigned32", "us", "*", ""),
    (159, "flowEndDeltaMicroseconds", "unsigned32", "us", "*", ""),
    (160, "systemInitTimeMilliseconds", "dateTimeMilliseconds", "s", "*", ""),
    (161, "flowDurationMilliseconds", "unsigned32", "ms", "*", ""),
    (162, "flowDurationMicroseconds", "unsigned32", "us", "*", ""),
    (163, "observedFlowTotalCount", "unsigned64", "flows", "", ""),
    (164, "ignoredPacketTotalCount", "unsigned64", "packets", "", ""),
    (165, "ignoredOctetTotalCount", "unsigned64", "octets", "", ""),
    (166, "notSentFlowTotalCount", "unsigned64", "flows", "", ""),
    (167, "notSentPacketTotalCount", "unsigned64", "packets", "", ""),
    (168, "notSentOctetTotalCount", "unsigned64", "octets", "", ""),
    (169, "destinationIPv6Prefix", "ipv6Address", None, "", ""),
    (170, "sourceIPv6Prefix", "ipv6Address", None, "", ""),
    (171, "postOctetTotalCount", "unsigned64", "octets", "", ""),
    (172, "postPacketTotalCount", "unsigned64", "packets", "", ""),
    (173, "flowKeyIndicator", "unsigned64", None, "", ""),
    (174, "postMCastPacketTotalCount", "unsigned64", "packets", "", ""),
    (175, "postMCastOctetTotalCount", "unsigned64", "octets", "", ""),
    (176, "icmpTypeIPv4", "unsigned8", None, "proto.icmp.type", "count"),
    (177, "icmpCodeIPv4", "unsigned8", None, "proto.icmp.code", "count"),
    (178, "icmpTypeIPv6", "unsigned8", None, "proto.icmp.type", "count"),
    (179, "icmpCodeIPv6", "unsigned8", None, "proto.icmp.code", "count"),
    (180, "udpSourcePort", "unsigned16", None, "flow.src_port", "port"),
    (181, "udpDestinationPort", "unsigned16", None, "flow.dst_port", "port"),
    (182, "tcpSourcePort", "unsigned16", None, "flow.src_port", "port"),
    (183, "tcpDestinationPort", "unsigned16", None, "flow.dst_port", "port"),
    (184, "tcpSequenceNumber", "unsigned32", None, "", ""),
    (185, "tcpAcknowledgementNumber", "unsigned32", None, "", ""),
    (186, "tcpWindowSize", "unsigned16", "octets", "", ""),
    (187, "tcpUrgentPointer", "unsigned16", None, "", ""),
    (188, "tcpHeaderLength", "unsigned8", "octets", "", ""),
    (189, "ipHeaderLength", "unsigned8", "octets", "", ""),
    (190, "totalLengthIPv4", "unsigned16", "octets", "", ""),
    (191, "payloadLengthIPv6", "unsigned16", "octets", "", ""),
    (192, "ipTTL", "unsigned8", "hops", "", ""),
    (193, "nextHeaderIPv6", "unsigned8", None, "", ""),
    (194, "mplsPayloadLength", "unsigned32", "octets", "", ""),
    (195, "ipDiffServCodePoint", "unsigned8", None, "", ""),
    (196, "ipPrecedence", "unsigned8", None, "", ""),
    (197, "fragmentFlags", "unsigned8", None, "", ""),
    (198, "octetDeltaSumOfSquares", "unsigned64", None, "", ""),
    (199, "octetTotalSumOfSquares", "unsigned64", None, "", ""),
    (200, "mplsTopLabelTTL", "unsigned8", "hops", "", ""),
    (201, "mplsLabelStackLength", "unsigned32", "octets", "", ""),
    (202, "mplsLabelStackDepth", "unsigned32", "label stack entries", "", ""),
    (203, "mplsTopLabelExp", "unsigned8", None, "", ""),
    (204, "ipPayloadLength", "unsigned32", "octets", "", ""),
    (205, "udpMessageLength", "unsigned16", "octets", "", ""),
    (206, "isMulticast", "unsigned8", None, "", ""),
    (207, "ipv4IHL", "unsigned8", "4 octets", "", ""),
    (208, "ipv4Options", "unsigned32", None, "", ""),
    (209, "tcpOptions", "unsigned64", None, "", ""),
    (210, "paddingOctets", "octetArray", None, "", ""),
    (211, "collectorIPv4Address", "ipv4Address", None, "", ""),
    (212, "collectorIPv6Address", "ipv6Address", None, "", ""),
    (213, "exportInterface", "unsigned32", None, "", ""),
    (214, "exportProtocolVersion", "unsigned8", None, "", ""),
    (215, "exportTransportProtocol", "unsigned8", None, "", ""),
    (216, "collectorTransportPort", "unsigned16", None, "", ""),
    (217, "exporterTransportPort", "unsigned16", None, "", ""),
    (218, "tcpSynTotalCount", "unsigned64", "packets", "flow.flag_count.syn", "count"),
    (219, "tcpFinTotalCount", "unsigned64", "packets", "flow.flag_count.fin", "count"),
    (220, "tcpRstTotalCount", "unsigned64", "packets", "flow.flag_count.rst", "count"),
    (221, "tcpPshTotalCount", "unsigned64", "packets", "flow.flag_count.psh", "count"),
    (222, "tcpAckTotalCount", "unsigned64", "packets", "flow.flag_count.ack", "count"),
    (223, "tcpUrgTotalCount", "unsigned64", "packets", "flow.flag_count.urg", "count"),
    (224, "ipTotalLength", "unsigned64", "octets", "", ""),
    (225, "postNATSourceIPv4Address", "ipv4Address", None, "", ""),
    (226, "postNATDestinationIPv4Address", "ipv4Address", None, "", ""),
    (227, "postNAPTSourceTransportPort", "unsigned16", None, "", ""),
    (228, "postNAPTDestinationTransportPort", "unsigned16", None, "", ""),
    (229, "natOriginatingAddressRealm", "unsigned8", None, "", ""),
    (230, "natEvent", "unsigned8", None, "", ""),
    (231, "initiatorOctets", "unsigned64", "octets", "*", ""),
    (232, "responderOctets", "unsigned64", "octets", "*", ""),
    (233, "firewallEvent", "unsigned8", None, "event.action", "firewall_event"),
    (234, "ingressVRFID", "unsigned32", None, "", ""),
    (235, "egressVRFID", "unsigned32", None, "", ""),
    (236, "VRFname", "string", None, "", ""),
    (237, "postMplsTopLabelExp", "unsigned8", None, "", ""),
    (238, "tcpWindowScale", "unsigned16", None, "", ""),
    (239, "biflowDirection", "unsigned8", None, "", ""),
    (240, "ethernetHeaderLength", "unsigned8", "octets", "", ""),
    (241, "ethernetPayloadLength", "unsigned16", "octets", "", ""),
    (242, "ethernetTotalLength", "unsigned16", "octets", "", ""),
    (243, "dot1qVlanId", "unsigned16", None, "flow.vlan", "count"),
    (244, "dot1qPriority", "unsigned8", None, "", ""),
    (245, "dot1qCustomerVlanId", "unsigned16", None, "flow.vlan_inner", "count"),
    (246, "dot1qCustomerPriority", "unsigned8", None, "", ""),
    (247, "metroEvcId", "string", None, "", ""),
    (248, "metroEvcType", "unsigned8", None, "", ""),
    (249, "pseudoWireId", "unsigned32", None, "", ""),
    (250, "pseudoWireType", "unsigned16", None, "", ""),
    (251, "pseudoWireControlWord", "unsigned32", None, "", ""),
    (252, "ingressPhysicalInterface", "unsigned32", None, "", ""),
    (253, "egressPhysicalInterface", "unsigned32", None, "", ""),
    (254, "postDot1qVlanId", "unsigned16", None, "", ""),
    (255, "postDot1qCustomerVlanId", "unsigned16", None, "", ""),
    (256, "ethernetType", "unsigned16", None, "", ""),
    (257, "postIpPrecedence", "unsigned8", None, "", ""),
    (258, "collectionTimeMilliseconds", "dateTimeMilliseconds", "s", "", ""),
    (259, "exportSctpStreamId", "unsigned16", None, "", ""),
    (260, "maxExportSeconds", "dateTimeSeconds", "s", "", ""),
    (261, "maxFlowEndSeconds", "dateTimeSeconds", "s", "", ""),
    (262, "messageMD5Checksum", "octetArray", None, "", ""),
    (263, "messageScope", "unsigned8", None, "", ""),
    (264, "minExportSeconds", "dateTimeSeconds", "s", "", ""),
    (265, "minFlowStartSeconds", "dateTimeSeconds", "s", "", ""),
    (266, "opaqueOctets", "octetArray", None, "", ""),
    (267, "sessionScope", "unsigned8", None, "", ""),
    (268, "maxFlowEndMicroseconds", "dateTimeMicroseconds", "s", "", ""),
    (269, "maxFlowEndMilliseconds", "dateTimeMilliseconds", "s", "", ""),
    (270, "maxFlowEndNanoseconds", "dateTimeNanoseconds", "s", "", ""),
    (271, "minFlowStartMicroseconds", "dateTimeMicroseconds", "s", "", ""),
    (272, "minFlowStartMilliseconds", "dateTimeMilliseconds", "s", "", ""),
    (273, "minFlowStartNanoseconds", "dateTimeNanoseconds", "s", "", ""),
    (274, "collectorCertificate", "octetArray", None, "", ""),
    (275, "exporterCertificate", "octetArray", None, "", ""),
    (276, "dataRecordsReliability", "boolean", None, "", ""),
    (277, "observationPointType", "unsigned8", None, "", ""),
    (278, "newConnectionDeltaCount", "unsigned32", None, "", ""),
    (279, "connectionSumDurationSeconds", "unsigned64", "s", "", ""),
    (280, "connectionTransactionId", "unsigned64", None, "", ""),
    (281, "postNATSourceIPv6Address", "ipv6Address", None, "", ""),
    (282, "postNATDestinationIPv6Address", "ipv6Address", None, "", ""),
    (283, "natPoolId", "unsigned32", None, "", ""),
    (284, "natPoolName", "string", None, "", ""),
    (285, "anonymizationFlags", "unsigned16", None, "", ""),
    (286, "anonymizationTechnique", "unsigned16", None, "", ""),
    (287, "informationElementIndex", "unsigned16", None, "", ""),
    (288, "p2pTechnology", "string", None, "", ""),
    (289, "tunnelTechnology", "string", None, "", ""),
    (290, "encryptedTechnology", "string", None, "", ""),
    (291, "basicList", "basicList", None, "", ""),
    (292, "subTemplateList", "subTemplateList", None, "", ""),
    (293, "subTemplateMultiList", "subTemplateMultiList", None, "", ""),
    (294, "bgpValidityState", "unsigned8", None, "", ""),
    (295, "IPSecSPI", "unsigned32", None, "", ""),
    (296, "greKey", "unsigned32", None, "", ""),
    (297, "natType", "unsigned8", None, "", ""),
    (298, "initiatorPackets", "unsigned64", "packets", "*", ""),
    (299, "responderPackets", "unsigned64", "packets", "*", ""),
    (300, "observationDomainName", "string", None, "", ""),
    (301, "selectionSequenceId", "unsigned64", None, "", ""),
    (302, "selectorId", "unsigned64", None, "", ""),
    (303, "informationElementId", "unsigned16", None, "", ""),
    (304, "selectorAlgorithm", "unsigned16", None, "", ""),
    (305, "samplingPacketInterval", "unsigned32", "packets", "flow.sampling_rate", "count"),
    (306, "samplingPacketSpace", "unsigned32", "packets", "", ""),
    (307, "samplingTimeInterval", "unsigned32", "us", "", ""),
    (308, "samplingTimeSpace", "unsigned32", "us", "", ""),
    (309, "samplingSize", "unsigned32", "packets", "", ""),
    (310, "samplingPopulation", "unsigned32", "packets", "", ""),
    (311, "samplingProbability", "float64", None, "", ""),
    (312, "dataLinkFrameSize", "unsigned16", "octets", "", ""),
    (313, "ipHeaderPacketSection", "octetArray", None, "", ""),
    (314, "ipPayloadPacketSection", "octetArray", None, "", ""),
    (315, "dataLinkFrameSection", "octetArray", None, "", ""),
    (316, "mplsLabelStackSection", "octetArray", None, "", ""),
    (317, "mplsPayloadPacketSection", "octetArray", None, "", ""),
    (318, "selectorIdTotalPktsObserved", "unsigned64", "packets", "", ""),
    (319, "selectorIdTotalPktsSelected", "unsigned64", "packets", "", ""),
    (320, "absoluteError", "float64", None, "", ""),
    (321, "relativeError", "float64", None, "", ""),
    (322, "observationTimeSeconds", "dateTimeSeconds", "s", "", ""),
    (323, "observationTimeMilliseconds", "dateTimeMilliseconds", "s", "", ""),
    (324, "observationTimeMicroseconds", "dateTimeMicroseconds", "s", "", ""),
    (325, "observationTimeNanoseconds", "dateTimeNanoseconds", "s", "", ""),
    (326, "digestHashValue", "unsigned64", None, "", ""),
    (327, "hashIPPayloadOffset", "unsigned64", None, "", ""),
    (328, "hashIPPayloadSize", "unsigned64", None, "", ""),
    (329, "hashOutputRangeMin", "unsigned64", None, "", ""),
    (330, "hashOutputRangeMax", "unsigned64", None, "", ""),
    (331, "hashSelectedRangeMin", "unsigned64", None, "", ""),
    (332, "hashSelectedRangeMax", "unsigned64", None, "", ""),
    (333, "hashDigestOutput", "boolean", None, "", ""),
    (334, "hashInitialiserValue", "unsigned64", None, "", ""),
    (335, "selectorName", "string", None, "", ""),
    (336, "upperCILimit", "float64", None, "", ""),
    (337, "lowerCILimit", "float64", None, "", ""),
    (338, "confidenceLevel", "float64", None, "", ""),
)

IE_BY_ID: dict[int, tuple[str, str, str | None]] = {i: (n, t, u) for i, n, t, u, _, _ in IPFIX_IES}
IE_BY_NAME: dict[str, int] = {n: i for i, n, *_ in IPFIX_IES}

#: Private enterprise number of RFC 5103 reverse information elements.
REVERSE_PEN = 29305
#: Reverse elements with a shared target (RFC 5103 section 6): element id -> (target, converter).
REVERSE_TARGETS: dict[int, tuple[str, str]] = {
    5: ("flow.ip_tos_bwd", "count"),
    6: ("flow.tcp_flags_bwd", "tcp_bits8"),
}
#: Elements whose reverse form is decoded (the rest of the reverse space is retained raw).
REVERSE_IES: tuple[int, ...] = (1, 2, 5, 6, 25, 26, 52, 53, 85, 86, 136, 150, 151, 152, 153, 154, 155, 156, 157,
                                161, 162, 218, 219, 220, 221, 222, 223, 231, 298)


def _ie_row(ie: int, name: str, abstract: str, unit: str | None, target: str, conv: str, *, reverse: bool = False) -> Row:
    dtype = ABSTRACT_DTYPES[abstract]
    native = f"ipfix.{'reverse.' if reverse else ''}{name}"
    note = f"IE {ie}" + (f" (reverse, PEN {REVERSE_PEN})" if reverse else "") + f", {abstract}."
    if target in ("", "*"):
        return R(("reverse." if reverse else "") + name, dtype, None, "", unit=unit, native=native, note=note)
    return R(("reverse." if reverse else "") + name, dtype, target, conv, unit=unit, note=note)


def _ie_rows() -> tuple[Row, ...]:
    rows = [_ie_row(i, n, t, u, tgt, cv) for i, n, t, u, tgt, cv in IPFIX_IES]
    for ie in REVERSE_IES:
        n, t, u = IE_BY_ID[ie]
        tgt, cv = REVERSE_TARGETS.get(ie, ("", ""))
        rows.append(_ie_row(ie, n, t, u, tgt, cv, reverse=True))
    return tuple(rows)


_FLOW_NOTES = (
    "Counters: octetTotalCount / packetTotalCount give flow.bytes_fwd / packets_fwd directly. Counts since the "
    "previous report (octetDeltaCount, packetDeltaCount, initiatorOctets / responderOctets as layer-4 payload, "
    "initiatorPackets / responderPackets, and the v5 dOctets / dPkts) are accumulated per flow across "
    "active-timeout exports into running totals (bounded cache; LOW_RELIABILITY with the observed share of the "
    "flow's lifetime when exports before collection began may be missing, AS-696).",
    "Reverse elements (RFC 5103) give the responder direction: reverse octets / packets to flow.bytes_bwd / "
    "packets_bwd, reverse tcpControlBits to flow.tcp_flags_bwd. Without them the backward fields stay NOT_SUPPLIED.",
    "Times: flow.start_time / end_time from the most precise element present (nanoseconds, microseconds, "
    "milliseconds, seconds, sysUpTime relative to the header, delta microseconds relative to the export time); "
    "event time = flow end; flow.duration = end - start, else flowDurationMicroseconds / Milliseconds.",
    "Derived: icmpTypeCodeIPv4 / IPv6 split into proto.icmp.type (high byte) and proto.icmp.code (low byte); "
    "flow.tcp_flags = tcpControlBits & 0x3F (of both directions); flow.packets_total; flow.unanswered when the "
    "reverse packet count is present; flow.bidir_ratio from IP-layer totals.",
    "Sampling: flow.sampling_rate from samplingInterval, samplerRandomInterval or samplingPacketInterval in the "
    "record, else from an options record of the same exporter and observation domain (sampler table).",
)

V5 = RecordMap(
    "netflow", "v5", "NetFlow v5 record", Level.FLOW, (
        R("exporter", "a", conv="addr", kind="id", note="Address the datagram came from."),
        R("version", "i", conv="count"),
        R("count", "i", conv="count"),
        R("sys_uptime", "i", conv="count", unit="ms"),
        R("unix_secs", "i", "@time", "epoch_seconds", unit="s"),
        R("unix_nsecs", "i", conv="count", unit="ns"),
        R("flow_sequence", "i", conv="count"),
        R("engine_type", "i", conv="count"),
        R("engine_id", "i", conv="count"),
        R("sampling_interval", "i", "flow.sampling_rate", "v5_sampling", also="native",
          note="Two mode bits and a 14-bit interval."),
        R("srcaddr", "a", "flow.src_ip", "addr"),
        R("dstaddr", "a", "flow.dst_ip", "addr"),
        R("nexthop", "a", "flow.next_hop", "addr"),
        R("input", "i", "flow.ingress_if", "count"),
        R("output", "i", "flow.egress_if", "count"),
        R("dPkts", "i", unit="packets", note="Packets of the flow since its last export (accumulated)."),
        R("dOctets", "i", unit="octets", note="Layer-3 octets since the last export (accumulated)."),
        R("first", "i", unit="ms", note="sysUptime at the flow's first packet."),
        R("last", "i", unit="ms", note="sysUptime at the flow's last packet."),
        R("srcport", "i", "flow.src_port", "port"),
        R("dstport", "i", "flow.dst_port", "port"),
        R("pad1", "i", conv="count"),
        R("tcp_flags", "i", "flow.tcp_flags_fwd", "count"),
        R("prot", "i", "flow.protocol", "count"),
        R("tos", "i", "flow.ip_tos", "count"),
        R("src_as", "i", "flow.src_as", "count"),
        R("dst_as", "i", "flow.dst_as", "count"),
        R("src_mask", "i", conv="count", unit="bits"),
        R("dst_mask", "i", conv="count", unit="bits"),
        R("pad2", "i", conv="count"),
    ), REF_V5, ocsf_class=4001,
    notes=(
        "Each record is one direction of a flow: the backward fields stay NOT_SUPPLIED.",
        "Times: start = unix_secs + unix_nsecs / 1e9 - (sys_uptime - first) / 1000 (32-bit wrap corrected); end "
        "likewise from last; event time = end. Header fields are repeated on every record of the datagram.",
        _FLOW_NOTES[0],
    ),
)

def _header_v9(n: str) -> tuple[Row, ...]:
    return (
        R("exporter", "a", conv="addr", kind="id", native=n + "exporter", note="Address the datagram came from."),
        R("version", "i", conv="count", native=n + "version"),
        R("count", "i", conv="count", native=n + "count"),
        R("sys_uptime", "i", conv="count", unit="ms", native=n + "sys_uptime"),
        R("unix_secs", "i", "@time", "epoch_seconds", unit="s", note="Export time; the event time is the flow end."),
        R("sequence", "i", conv="count", native=n + "sequence"),
        R("source_id", "i", conv="count", native=n + "source_id"),
        R("template_id", "i", conv="count", native=n + "template_id"),
        R("scope_fields", "S", conv="list", native=n + "scope_fields", note="Options records: names of the scope fields."),
    )


def _header_ipfix(n: str) -> tuple[Row, ...]:
    return (
        R("exporter", "a", conv="addr", kind="id", native=n + "exporter", note="Address the message came from."),
        R("version", "i", conv="count", native=n + "version"),
        R("length", "i", conv="count", unit="bytes", native=n + "message_length"),
        R("export_time", "i", "@time", "epoch_seconds", unit="s", note="Export time; the event time is the flow end."),
        R("sequence", "i", conv="count", native=n + "sequence"),
        R("observation_domain_id", "i", conv="count", native=n + "observation_domain_id"),
        R("template_id", "i", conv="count", native=n + "template_id"),
        R("scope_fields", "S", conv="list", native=n + "scope_fields", note="Options records: names of the scope fields."),
    )


_OPTIONS_NOTES = (
    "An options data record describes the exporter (sampler, interface, metering process): it updates the sampler "
    "table of its exporter and observation domain and becomes a state update with the exporter as subject.",
)

V9 = RecordMap("netflow", "v9", "NetFlow v9 data record", Level.FLOW, (*_header_v9("netflow.v9."), *_ie_rows()), REF_V9,
               ocsf_class=4001, notes=_FLOW_NOTES)
V9_OPTIONS = RecordMap("netflow", "v9_options", "NetFlow v9 options data record", Level.DEVICE,
                       (*_header_v9("netflow.v9."), *_ie_rows()), REF_V9, ocsf_class=0, notes=_OPTIONS_NOTES)
IPFIX = RecordMap("ipfix", "data", "IPFIX data record", Level.FLOW, (*_header_ipfix("ipfix.msg."), *_ie_rows()), REF_IPFIX,
                  ocsf_class=4001, notes=_FLOW_NOTES)
IPFIX_OPTIONS = RecordMap("ipfix", "options", "IPFIX options data record", Level.DEVICE,
                          (*_header_ipfix("ipfix.msg."), *_ie_rows()), REF_IPFIX, ocsf_class=0, notes=_OPTIONS_NOTES)

# sFlow v5.
_SFLOW_DATAGRAM: tuple[Row, ...] = (
    R("agent_address", "a", "event.hostname", "addr", note="Agent (switch or router) address: the subject or observer."),
    R("sub_agent_id", "i", conv="count"),
    R("datagram_sequence", "i", conv="count"),
    R("uptime", "i", "dev.uptime", "ms", unit="s"),
    R("source_id_type", "i", conv="count", note="0 ifIndex, 1 smonVlanDataSource, 2 entPhysicalEntry."),
    R("source_id_index", "i", conv="count"),
    R("sample_sequence", "i", conv="count"),
)

SFLOW_FLOW = RecordMap(
    "sflow", "flow_sample", "sFlow v5 flow sample (formats 1 and 3)", Level.PACKET, (
        *_SFLOW_DATAGRAM,
        R("sampling_rate", "i", "flow.sampling_rate", "count"),
        R("sample_pool", "i", conv="count", unit="packets"),
        R("drops", "i", conv="count", unit="packets", note="Cumulative drops for lack of resources."),
        R("input", "i", "flow.ingress_if", "sflow_if"),
        R("output", "i", "flow.egress_if", "sflow_if"),
        R("header_protocol", "i", conv="count", note="1 Ethernet, 11 IPv4, 12 IPv6 (sampled header)."),
        R("frame_length", "i", conv="count", unit="bytes"),
        R("stripped", "i", conv="count", unit="bytes"),
        R("header", "y", note="Sampled header bytes; decoded with dpkt."),
        R("eth_length", "i", conv="count", unit="bytes"),
        R("eth_src", "m", "flow.src_mac", "mac"),
        R("eth_dst", "m", "flow.dst_mac", "mac"),
        R("eth_type", "i", conv="count"),
        R("ip_length", "i", conv="count", unit="bytes"),
        R("ip_protocol", "i", "flow.protocol", "count"),
        R("ip_src", "a", "flow.src_ip", "addr"),
        R("ip_dst", "a", "flow.dst_ip", "addr"),
        R("ip_src_port", "i", "flow.src_port", "port"),
        R("ip_dst_port", "i", "flow.dst_port", "port"),
        R("ip_tcp_flags", "i", "flow.tcp_flags_fwd", "count"),
        R("ip_tos", "i", "flow.ip_tos", "count"),
        R("src_vlan", "i", "flow.vlan", "count"),
        R("src_priority", "i", conv="count"),
        R("dst_vlan", "i", conv="count"),
        R("dst_priority", "i", conv="count"),
        R("nexthop", "a", "flow.next_hop", "addr"),
        R("src_mask_len", "i", conv="count", unit="bits"),
        R("dst_mask_len", "i", conv="count", unit="bits"),
        R("gateway_as", "i", conv="count"),
        R("src_as", "i", "flow.src_as", "count"),
        R("src_peer_as", "i", conv="count"),
        R("dst_as_path", "I", conv="list_int"),
        R("communities", "I", conv="list_int"),
        R("localpref", "i", conv="count"),
        R("src_user", "s", kind="id"),
        R("dst_user", "s", kind="id"),
        R("url_direction", "i", conv="count"),
        R("url", "s", kind="id"),
        R("url_host", "s", kind="id"),
    ), REF_SFLOW, ocsf_class=4001,
    notes=(
        "One state update per flow sample: the sampled packet's header is decoded (dpkt) for addresses, ports, "
        "protocol, TTL, TOS, TCP flags, IP total length and DNS fields; Ethernet, IPv4 and IPv6 data records fill "
        "the same fields when there is no sampled header.",
        "Estimates (AS-697): flow.packets_fwd = sampling_rate and flow.bytes_fwd = IP total length x sampling_rate, "
        "LOW_RELIABILITY with reliability 1 / (1 + 1.96) (one sample; sFlow sampling accuracy 196 x sqrt(1/c) "
        "percent at 95 %). Values of the one packet (TTL, flags, TOS, ports) are OBSERVED.",
        "dev.capture_dropped: delta of drops between consecutive samples of the same agent and source.",
    ),
)

SFLOW_COUNTERS = RecordMap(
    "sflow", "counter_sample", "sFlow v5 counter sample (formats 2 and 4)", Level.DEVICE, (
        *_SFLOW_DATAGRAM,
        R("ifIndex", "i", "dev.if_index", "count"),
        R("ifType", "i", "dev.if_type", "count"),
        R("ifSpeed", "i", "dev.if_speed", "double", unit="bit/s"),
        R("ifDirection", "i", conv="count", note="0 unknown, 1 full duplex, 2 half duplex, 3 in, 4 out."),
        R("ifStatus", "i", "dev.if_status", "count"),
        *(R(n, "i", conv="count", unit=u, note="Cumulative counter.") for n, u in (
            ("ifInOctets", "bytes"), ("ifInUcastPkts", "packets"), ("ifInMulticastPkts", "packets"),
            ("ifInBroadcastPkts", "packets"), ("ifInDiscards", "packets"), ("ifInErrors", "packets"),
            ("ifInUnknownProtos", "packets"), ("ifOutOctets", "bytes"), ("ifOutUcastPkts", "packets"),
            ("ifOutMulticastPkts", "packets"), ("ifOutBroadcastPkts", "packets"), ("ifOutDiscards", "packets"),
            ("ifOutErrors", "packets"))),
        R("ifPromiscuousMode", "i", "dev.if_promiscuous", "truth_value", note="TruthValue: 1 true, 2 false."),
        *(R(n, "i", conv="count", unit="frames") for n in (
            "dot3StatsAlignmentErrors", "dot3StatsFCSErrors", "dot3StatsSingleCollisionFrames",
            "dot3StatsMultipleCollisionFrames", "dot3StatsSQETestErrors", "dot3StatsDeferredTransmissions",
            "dot3StatsLateCollisions", "dot3StatsExcessiveCollisions", "dot3StatsInternalMacTransmitErrors",
            "dot3StatsCarrierSenseErrors", "dot3StatsFrameTooLongs", "dot3StatsInternalMacReceiveErrors",
            "dot3StatsSymbolErrors")),
        R("vlan_id", "i", conv="count"),
        R("vlan_octets", "i", conv="count", unit="bytes"),
        R("vlan_ucastPkts", "i", conv="count", unit="packets"),
        R("vlan_multicastPkts", "i", conv="count", unit="packets"),
        R("vlan_broadcastPkts", "i", conv="count", unit="packets"),
        R("vlan_discards", "i", conv="count", unit="packets"),
        R("cpu_5s", "i", conv="count", unit="%/100"),
        R("cpu_1m", "i", conv="count", unit="%/100"),
        R("cpu_5m", "i", conv="count", unit="%/100"),
        R("total_memory", "i", conv="count", unit="bytes"),
        R("free_memory", "i", conv="count", unit="bytes"),
    ), REF_SFLOW, ocsf_class=0,
    notes=(
        "Counter deltas between consecutive samples of the same agent and source (32-bit counters corrected for "
        "one wrap; a decrease beyond that is a reset and gives NOT_SUPPLIED): dev.if_in/out_octets, "
        "dev.if_in/out_packets (unicast + multicast + broadcast), dev.if_errors and dev.if_discards (in + out), "
        "the per-direction components, dev.interval from the agent uptime.",
        "Entities: the agent as subject.",
    ),
)

FLOWEXPORT_MAPS: tuple[RecordMap, ...] = (V5, V9, V9_OPTIONS, IPFIX, IPFIX_OPTIONS, SFLOW_FLOW, SFLOW_COUNTERS)
