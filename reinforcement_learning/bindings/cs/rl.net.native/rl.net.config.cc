#include "rl.net.config.h"

API u::configuration* CreateConfig()
{
    return new u::configuration();
}

API void DeleteConfig(u::configuration* config)
{
    delete config;
}

API int LoadConfigFromJson(const int length, const char* json, u::configuration* config)
{
  // This is a deep copy, so it is safe to push a pinned-managed string here.
    const std::string json_str (json, length);

    return cfg::create_from_json(json_str, *config); // TODO: API Status
}