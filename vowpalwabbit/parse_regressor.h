// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once
#include <string>
#include "global_data.h"

// forward declaration
namespace VW
{
namespace config
{
struct options_i;
}
}  // namespace VW

void read_regressor_file(vw& all, std::vector<std::string> files, io_buf& io_temp);

void finalize_regressor(vw& all, std::string reg_name);
void initialize_regressor(vw& all);

void save_predictor(vw& all, std::string reg_name, size_t current_pass);
void save_load_header(
    vw& all, io_buf& model_file, bool read, bool text, std::string& file_options, VW::config::options_i& options);

void parse_mask_regressor_args(vw& all, std::string feature_mask, std::vector<std::string> initial_regressors);
