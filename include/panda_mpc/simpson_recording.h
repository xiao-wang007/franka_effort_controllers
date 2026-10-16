#pragma once

// Single-producer/single-consumer recorder. Push() is called by update(); all
// formatting, directory creation, flushing and file writes run outside
// update().
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace franka_torque_controller {
class SimpsonRecording {
public:
  static constexpr size_t kBase = 15, kJointGroups = 16,
                          kWidth = kBase + 7 * kJointGroups + 12;
  using Row = std::array<double, kWidth>;
  static std::vector<std::string> Columns() {
    std::vector<std::string> out = {"sequence",
                                    "stamp",
                                    "period",
                                    "controller_start",
                                    "trajectory_start",
                                    "trajectory_id",
                                    "execution_id",
                                    "elapsed",
                                    "reference_time",
                                    "duration",
                                    "holding",
                                    "torque_limit_mask",
                                    "rate_limit_mask",
                                    "dropped_samples",
                                    "active"};
    for (const auto *group :
         {"q", "dq_raw", "dq_filtered", "q_ref", "dq_ref", "dq_from_q_ref",
          "tau_ff", "tau_position", "tau_velocity", "tau_requested",
          "tau_clamped", "tau_command", "tau_J_d", "tau_J", "kp", "kd"})
      for (int j = 1; j <= 7; ++j)
        out.push_back(std::string(group) + std::to_string(j));
    for (const auto *group : {"p_ref", "v_ref", "p_measured", "v_measured"})
      for (const auto *axis : {"x", "y", "z"})
        out.push_back(std::string(group) + "_" + axis);
    return out;
  }
  static std::string JsonString(const std::string &s) {
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
      if (c == '"' || c == '\\')
        o << '\\' << c;
      else if (c < 32)
        o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
          << std::dec;
      else
        o << c;
    }
    o << '"';
    return o.str();
  }
  SimpsonRecording(const std::string &root, const std::string &settings)
      : rows_(new Row[kCapacity]) {
    std::string base = root;
    if (base == "~" || base.compare(0, 2, "~/") == 0) {
      const char *home = std::getenv("HOME");
      if (!home)
        throw std::runtime_error("HOME is unavailable for recording path");
      base.replace(0, 1, home);
    }
    if (base.empty())
      throw std::runtime_error("empty recording_directory");
    MakeDirectories(base);
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&seconds, &tm);
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                            now.time_since_epoch())
                            .count() %
                        1000000;
    std::ostringstream name;
    name << base << "/simpson_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S")
         << '_' << std::setw(6) << std::setfill('0') << micros << '_'
         << getpid();
    directory_ = name.str();
    if (::mkdir(directory_.c_str(), 0700) != 0)
      throw std::runtime_error("cannot create " + directory_);
    WriteFile(directory_ + "/settings.json", settings);
    samples_.open(directory_ + "/samples.csv");
    executions_.open(directory_ + "/executions.csv");
    if (!samples_ || !executions_)
      throw std::runtime_error("cannot open recording files");
    const auto names = Columns();
    for (size_t i = 0; i < names.size(); ++i)
      samples_ << (i ? "," : "") << names[i];
    samples_ << '\n' << std::setprecision(17);
    executions_ << "execution_id,trajectory_id,controller_start,trajectory_"
                   "start,first_sample_stamp,duration\n"
                << std::setprecision(17);
    worker_ = std::thread(&SimpsonRecording::Run, this);
  }
  ~SimpsonRecording() {
    running_.store(false, std::memory_order_release);
    if (worker_.joinable())
      worker_.join();
  }
  SimpsonRecording(const SimpsonRecording &) = delete;
  SimpsonRecording &operator=(const SimpsonRecording &) = delete;
  const std::string &directory() const { return directory_; }
  uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
  bool healthy() const { return healthy_.load(std::memory_order_relaxed); }
  // Only the control thread calls this method; never blocks or writes files.
  bool Push(const Row &row) {
    const auto w = write_.load(std::memory_order_relaxed);
    if (w - read_.load(std::memory_order_acquire) >= kCapacity || !healthy()) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    rows_[w % kCapacity] = row;
    write_.store(w + 1, std::memory_order_release);
    return true;
  }
  // Only called from the non-realtime trajectory subscriber callback.
  void SaveReference(uint64_t id, const std::string &csv,
                     const std::string &metadata) {
    const auto stem = directory_ + "/reference_" + std::to_string(id);
    WriteFile(stem + ".csv", csv);
    WriteFile(stem + ".json", metadata);
  }

private:
  static constexpr uint64_t kCapacity = 8192;
  static void MakeDirectories(const std::string &path) {
    for (size_t i = 1; i <= path.size(); ++i) {
      if (i < path.size() && path[i] != '/')
        continue;
      const auto part = path.substr(0, i);
      struct stat st {};
      if (::mkdir(part.c_str(), 0700) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create recording directory " + part);
      if (::stat(part.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
        throw std::runtime_error("recording path is not a directory: " + part);
    }
  }
  static void WriteFile(const std::string &path, const std::string &value) {
    std::ofstream out(path);
    out << value;
    out.close();
    if (!out)
      throw std::runtime_error("failed writing " + path);
  }
  void Run() {
    uint64_t previous_execution = 0, saved_drops = 0;
    auto last_flush = std::chrono::steady_clock::now();
    while (running_.load(std::memory_order_acquire) ||
           read_.load(std::memory_order_relaxed) !=
               write_.load(std::memory_order_acquire)) {
      auto r = read_.load(std::memory_order_relaxed);
      const auto w = write_.load(std::memory_order_acquire);
      const auto end = std::min<uint64_t>(w, r + 512);
      for (; r < end; ++r) {
        const auto &row = rows_[r % kCapacity];
        if (row[6] != previous_execution) {
          previous_execution = static_cast<uint64_t>(row[6]);
          executions_ << row[6] << ',' << row[5] << ',' << row[3] << ','
                      << row[4] << ',' << row[1] << ',' << row[9] << '\n';
        }
        for (size_t i = 0; i < row.size(); ++i)
          samples_ << (i ? "," : "") << row[i];
        samples_ << '\n';
      }
      read_.store(r, std::memory_order_release);
      if (std::chrono::steady_clock::now() - last_flush >
          std::chrono::milliseconds(500)) {
        samples_.flush();
        executions_.flush();
        if (!samples_ || !executions_) {
          healthy_.store(false, std::memory_order_relaxed);
          std::cerr << "Simpson recording write failed in " << directory_
                    << std::endl;
        }
        if (dropped() != saved_drops) {
          saved_drops = dropped();
          std::cerr << "Simpson recording dropped " << saved_drops
                    << " samples in " << directory_ << std::endl;
        }
        last_flush = std::chrono::steady_clock::now();
      }
      if (r == w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    samples_.close();
    executions_.close();
    const bool ok = healthy() && !samples_.fail() && !executions_.fail();
    try {
      WriteFile(directory_ + "/status.json",
                "{\"dropped_samples\":" + std::to_string(dropped()) +
                    ",\"written_samples\":" + std::to_string(read_.load()) +
                    ",\"write_ok\":" + (ok ? "true" : "false") + "}\n");
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
  }
  std::unique_ptr<Row[]> rows_;
  std::atomic<uint64_t> write_{0}, read_{0}, dropped_{0};
  std::atomic<bool> running_{true}, healthy_{true};
  std::string directory_;
  std::ofstream samples_, executions_;
  std::thread worker_;
};
} // namespace franka_torque_controller
