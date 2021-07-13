// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once
#include <cstdint>
#include <cstddef>

constexpr int quadratic_constant = 27942141;
constexpr int cubic_constant = 21791;
constexpr int cubic_constant2 = 37663;
constexpr int affix_constant = 13903957;
constexpr uint64_t constant = 11650396;

constexpr float probability_tolerance = 1e-5f;

// FNV-like hash constant for 32bit
// http://www.isthe.com/chongo/tech/comp/fnv/#FNV-param
constexpr uint32_t FNV_prime = 16777619;

constexpr unsigned char default_namespace = 32;
// This is a special case, to maintain legacy behavior the parser uses a channel hash of zero for default ns.
constexpr uint64_t DEFAULT_NAMESPACE_HASH = 0;
constexpr unsigned char wildcard_namespace = 58;  // :
constexpr unsigned char wap_ldf_namespace = 126;
constexpr unsigned char history_namespace = 127;
constexpr unsigned char constant_namespace = 128;
constexpr unsigned char nn_output_namespace = 129;
constexpr unsigned char autolink_namespace = 130;
constexpr unsigned char neighbor_namespace =
    131;  // this is \x83 -- to do quadratic, say "-q a`printf "\x83"` on the command line
constexpr unsigned char affix_namespace = 132;         // this is \x84
constexpr unsigned char spelling_namespace = 133;      // this is \x85
constexpr unsigned char conditioning_namespace = 134;  // this is \x86
constexpr unsigned char dictionary_namespace = 135;    // this is \x87
constexpr unsigned char node_id_namespace = 136;       // this is \x88
constexpr unsigned char baseline_enabled_message_namespace = 137;  // this is \x89
constexpr unsigned char ccb_slot_namespace = 139;
constexpr unsigned char ccb_id_namespace = 140;
constexpr unsigned char interact_reduction_namespace = 141;

typedef float weight;

constexpr size_t NUM_NAMESPACES = 256;

constexpr const char* CCB_LABEL = "ccb";
constexpr const char* SLATES_LABEL = "slates";
constexpr const char* SHARED_TYPE = "shared";
constexpr const char* ACTION_TYPE = "action";
constexpr const char* SLOT_TYPE = "slot";
constexpr const char* CA_LABEL = "ca";
constexpr const char* PDF = "pdf";
constexpr const char* CHOSEN_ACTION = "chosen_action";

static constexpr uint32_t SHARED_EX_INDEX = 0;
static constexpr uint32_t TOP_ACTION_INDEX = 0;
