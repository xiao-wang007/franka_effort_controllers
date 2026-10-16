#include <cassert>
#include <cstdio>
#include <limits>
#include <panda_mpc/simpson_recording.h>
using franka_torque_controller::SimpsonRecording;
static std::string Read(const std::string &p) {
  std::ifstream f(p);
  return std::string(std::istreambuf_iterator<char>(f), {});
}
static std::vector<std::string> Split(const std::string &s) {
  std::stringstream stream(s);
  std::vector<std::string> result;
  std::string cell;
  while (std::getline(stream, cell, ','))
    result.push_back(cell);
  return result;
}
int main() {
  static_assert(SimpsonRecording::kWidth == 139, "recording schema changed");
  assert(SimpsonRecording::Columns().size() == SimpsonRecording::kWidth);
  assert(SimpsonRecording::JsonString("x\n\"\\") == "\"x\\u000a\\\"\\\\\"");
  char temp[] = "/tmp/simpson-recording-test-XXXXXX";
  assert(mkdtemp(temp));
  std::string directory;
  size_t accepted = 0;
  {
    SimpsonRecording logger(std::string(temp) + "/nested",
                            "{\"alpha\":0.99}\n");
    directory = logger.directory();
    logger.SaveReference(9, "t,q1\n0,0\n", "{\"trajectory_id\":9}\n");
    for (int i = 1; i <= 20000; ++i) {
      SimpsonRecording::Row row;
      for (size_t j = 0; j < row.size(); ++j)
        row[j] = i * 1000 + j;
      row[0] = i;
      row[5] = 9;
      row[6] = i <= 10000 ? 1 : 2;
      row[138] = std::numeric_limits<double>::quiet_NaN();
      if (logger.Push(row))
        ++accepted;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    for (int i = 20001; i <= 21000; ++i) {
      SimpsonRecording::Row row;
      for (size_t j = 0; j < row.size(); ++j)
        row[j] = i * 1000 + j;
      row[0] = i;
      row[5] = 9;
      row[6] = 2;
      row[138] = std::numeric_limits<double>::quiet_NaN();
      if (logger.Push(row))
        ++accepted;
    }
    assert(logger.healthy());
    assert(logger.dropped() + accepted == 21000);
  } // Drain outstanding rows before destruction returns.
  std::ifstream csv(directory + "/samples.csv");
  std::string line;
  assert(std::getline(csv, line));
  assert(Split(line) == SimpsonRecording::Columns());
  size_t count = 0;
  double previous = 0;
  while (std::getline(csv, line)) {
    const auto cells = Split(line);
    assert(cells.size() == SimpsonRecording::kWidth);
    double seq = std::stod(cells[0]);
    assert(seq > previous);
    previous = seq;
    assert(std::stod(cells[1]) == seq * 1000 + 1);
    assert(std::stod(cells[5]) == 9);
    assert(std::stod(cells[137]) == seq * 1000 + 137);
    assert(std::isnan(std::stod(cells[138])));
    ++count;
  }
  assert(count == accepted);
  assert(Read(directory + "/reference_9.csv") == "t,q1\n0,0\n");
  assert(Read(directory + "/reference_9.json") == "{\"trajectory_id\":9}\n");
  assert(Read(directory + "/status.json")
             .find("\"written_samples\":" + std::to_string(accepted)) !=
         std::string::npos);
  assert(Read(directory + "/status.json").find("\"write_ok\":true") !=
         std::string::npos);
  assert(Read(directory + "/executions.csv").find("2,9,") != std::string::npos);
  std::ofstream(std::string(temp) + "/file") << "not a directory";
  bool rejected = false;
  try {
    SimpsonRecording invalid(std::string(temp) + "/file/child", "{}");
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  assert(rejected);
  std::cout << "Recorder schema, threaded overflow/wrap, references, execution "
               "IDs, flush and path failure passed: "
            << directory << '\n';
}
