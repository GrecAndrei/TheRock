// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace rxinfer::json {

struct Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    using Variant = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
    Variant data = nullptr;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    const Value* get(const std::string& key) const {
        if (!is_object()) return nullptr;
        const auto& obj = std::get<Object>(data);
        auto it = obj.find(key);
        return (it == obj.end()) ? nullptr : &it->second;
    }

    int64_t as_i64(int64_t def = 0) const {
        return is_number() ? static_cast<int64_t>(std::get<double>(data)) : def;
    }
    double as_f64(double def = 0.0) const {
        return is_number() ? std::get<double>(data) : def;
    }
    bool as_bool(bool def = false) const {
        return is_bool() ? std::get<bool>(data) : def;
    }
    std::string as_str(const std::string& def = "") const {
        return is_string() ? std::get<std::string>(data) : def;
    }
};

Value parse(const std::string& text);

} // namespace rxinfer::json
