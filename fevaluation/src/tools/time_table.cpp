#include <unistd.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

struct TimeBound {
  uint64_t opentime{0};
  uint64_t closetime{0};
  uint64_t marker{0};
};

uint64_t HumanReadableMicrosecond(const std::string& str);

string RevertTime(uint64_t usec); //HH:MM:SS
inline uint64_t convertTime(const char* timestr);  // HH:MM:SS.mmm

void to_file(std::vector<TimeBound>& m_bounds);

int main(int argc, char** argv) {
    bool is_crypto = false;
    bool force930 = false;
    int64_t itvl = HumanReadableMicrosecond("1s");
    std::vector<TimeBound> m_bounds;
    m_bounds.clear();
  if (is_crypto) {
    auto open_usec = convertTime("00:00:00");
    auto end_usec = convertTime("25:00:00");
    auto close_usec = open_usec + itvl;
    do {
      m_bounds.push_back({open_usec, close_usec, close_usec});
      open_usec = close_usec;
      close_usec += itvl;
    } while (close_usec <= end_usec);
    return 0;
  }

  auto open_usec = convertTime("09:20:00");
  auto close_usec = convertTime("09:26:00");
  auto marker_usec = force930 ? convertTime("09:29:59") : convertTime("09:30:00");
  auto end_marker_usec = convertTime("11:30:00");
  do {
    m_bounds.push_back({open_usec, close_usec, marker_usec});
    open_usec = marker_usec;
    marker_usec += itvl;
    close_usec = marker_usec;
  } while (marker_usec < end_marker_usec);

  open_usec = m_bounds.back().closetime;
  close_usec = convertTime("13:00:00");
  marker_usec = convertTime("13:00:00");
  end_marker_usec = convertTime("14:57:00");

  do {
    m_bounds.push_back({open_usec, close_usec, marker_usec});
    open_usec = marker_usec;
    marker_usec += itvl;
    close_usec = marker_usec;
  } while (marker_usec < end_marker_usec);

  m_bounds.push_back({m_bounds.back().closetime, convertTime("14:57:00"), convertTime("14:57:00")});
  m_bounds.push_back({m_bounds.back().closetime, convertTime("15:01:00"), convertTime("15:00:00")});

  to_file(m_bounds);
  return 0;
}

void to_file(std::vector<TimeBound>& m_bounds) {
    string file = "/tmp/m_bounds.txt";
    string filename{ file };

    //ofstream ofs(filename, ofstream::out | ofstream::app);
    ofstream ofs(filename, ofstream::out | ofstream::trunc);

    if (!ofs)
    {
        cout << "open file failed" << endl;
    }
    else
    {
        ofs << "opentime,closetime,marker" << endl;
        for (size_t i = 0; i < m_bounds.size(); i++)
        {
           ofs << RevertTime(m_bounds[i].opentime) << "," << RevertTime(m_bounds[i].closetime) << "," << RevertTime(m_bounds[i].marker) << endl;
        }
        ofs.close();
    }

    cout << "output to " << file << endl;
}

inline string RevertTime(uint64_t usec) //HHMMSSmmm
{
    int sec = usec / 1000000;
    int h   = sec / 3600;
    sec -= h * 3600;
    int m   = sec / 60;
    sec -= m * 60;
    //int time = h * 10000000 + m * 100000 + sec * 1000 + ms;
    std::ostringstream ss;
    ss << std::setw(2) << std::setfill('0') << h << ':' ;
    ss << std::setw(2) << std::setfill('0') << m << ":";
    ss << std::setw(2) << std::setfill('0') << sec;
    return ss.str();
}

inline uint64_t convertTime(const char* timestr)  // HH:MM:SS.mmm
{
  uint64_t h = (timestr[0] - '0') * 10 + (timestr[1] - '0');
  uint64_t m = (timestr[3] - '0') * 10 + (timestr[4] - '0');
  double s = atof(timestr + 6);
  uint64_t utime = static_cast<uint64_t>((h * 3600 + m * 60 + s) * 1000000);
  return utime;
}

uint64_t HumanReadableMicrosecond(const std::string& str) {
  constexpr uint64_t _second = 1000ull * 1000ull;
  constexpr uint64_t _minute = 60ull * _second;
  constexpr uint64_t _hour = 60ull * _minute;
  std::string str_ = str;
  std::transform(str_.begin(), str_.end(), str_.begin(), ::tolower);
  uint64_t interval = 0;
  if (str_.find("milliseconds") != std::string::npos) {
    interval = 1000ull * std::stoull(str_.substr(0, str_.find("milliseconds")));
  } else if (str_.find("millisecond") != std::string::npos) {
    interval = 1000ull * std::stoull(str_.substr(0, str_.find("millisecond")));
  } else if (str_.find("ms") != std::string::npos) {
    interval = 1000ull * std::stoull(str_.substr(0, str_.find("ms")));
  } else if (str_.find("minutes") != std::string::npos) {
    interval = _minute * std::stoull(str_.substr(0, str_.find("minutes")));
  } else if (str_.find("minute") != std::string::npos) {
    interval = _minute * std::stoull(str_.substr(0, str_.find("minute")));
  } else if (str_.find("min") != std::string::npos) {
    interval = _minute * std::stoull(str_.substr(0, str_.find("min")));
  } else if (str_.find("hours") != std::string::npos) {
    interval = _hour * std::stoull(str_.substr(0, str_.find("hours")));
  } else if (str_.find("hour") != std::string::npos) {
    interval = _hour * std::stoull(str_.substr(0, str_.find("hour")));
  } else if (str_.find("macroseconds") != std::string::npos) {
    interval = std::stoull(str_.substr(0, str_.find("macroseconds")));
  } else if (str_.find("macrosecond") != std::string::npos) {
    interval = std::stoull(str_.substr(0, str_.find("macrosecond")));
  } else if (str_.find("seconds") != std::string::npos) {
    interval = _second * std::stoull(str_.substr(0, str_.find("seconds")));
  } else if (str_.find("second") != std::string::npos) {
    interval = _second * std::stoull(str_.substr(0, str_.find("second")));
  } else if (str_.find("sec") != std::string::npos) {
    interval = _second * std::stoull(str_.substr(0, str_.find("sec")));
  } else if (str_.find("quarter") != std::string::npos) {
    interval = 15ull * _minute * std::stoull(str_.substr(0, str_.find("quarter")));
  } else if (str_.find("us") != std::string::npos) {
    interval = std::stoull(str_.substr(0, str_.find("us")));
  } else if (str_.find('q') != std::string::npos) {
    interval = 15ull * _minute * std::stoull(str_.substr(0, str_.find('q')));
  } else if (str_.find('m') != std::string::npos) {
    interval = _minute * std::stoull(str_.substr(0, str_.find('m')));
  } else if (str_.find('s') != std::string::npos) {
    interval = _second * std::stoull(str_.substr(0, str_.find('s')));
  } else if (str_.find('h') != std::string::npos) {
    interval = _hour * std::stoull(str_.substr(0, str_.find('h')));
  } else if (not str.empty()) {
    interval = std::stoull(str_) * 1000ull;
  }
  return interval;
}

