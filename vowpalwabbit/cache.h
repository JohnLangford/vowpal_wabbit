// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "v_array.h"
#include "io_buf.h"
#include "example.h"

char* run_len_decode(char* p, size_t& i);
char* run_len_encode(char* p, size_t i);

int read_cached_features(vw* all, v_array<example*>& examples);
void cache_tag(io_buf& cache, const v_array<char>& tag);
void cache_features(io_buf& cache, const example* ae, uint64_t mask);
void output_byte(io_buf& cache, unsigned char s);
void output_features(io_buf& cache, unsigned char index, uint64_t ns_hash, const features& fs, uint64_t mask);

namespace VW
{
uint32_t convert(size_t number);
// What is written by write_example_to_cache can be read by read_example_from_cache
void write_example_to_cache(io_buf& output, example* ae, label_parser& lbl_parser, uint64_t parse_mask);
int read_example_from_cache(
    io_buf& input, example* ae, label_parser& lbl_parser, bool sorted_cache, shared_data* shared_dat);
}  // namespace VW
