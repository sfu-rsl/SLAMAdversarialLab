// Differential harness (test build: -DSAL_DEADLINE_TEST_CLOCK).
// Reads traces from stdin, runs the C++ iterator under the injected virtual
// clock, prints the survivor+dropped sets. A Python driver runs the identical
// traces through the Python iterator and compares bit-for-bit.
//
// Trace line: n fps warmup queue policy k p0 p1 ... p(k-1)
//   policy: 0=drop_oldest, 1=drop_newest ; p_i = per-survivor consumer seconds.
#include "deadline_iterator.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

static void set_i(const char* k, long v){ setenv(k, std::to_string(v).c_str(), 1); }
static void set_d(const char* k, double v){ char b[64]; std::snprintf(b,sizeof b,"%.10g",v); setenv(k,b,1); }

int main(){
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    int n, warmup, queue, policy, k; double fps;
    ss >> n >> fps >> warmup >> queue >> policy >> k;
    std::vector<double> pt(k);
    for (int i=0;i<k;i++) ss >> pt[i];

    set_d("SAL_DEADLINE_FPS", fps);
    set_i("SAL_DEADLINE_WARMUP_FRAMES", warmup);
    set_i("SAL_DEADLINE_QUEUE_SIZE", queue);
    setenv("SAL_DEADLINE_DROP_POLICY", policy==0? "drop_oldest":"drop_newest", 1);
    unsetenv("SAL_DROP_LOG_PATH");
    unsetenv("SAL_PROGRESS_PATH");

    sal::DeadlineIterator::sal_reset_test_clock();
    sal::DeadlineIterator dl(n);
    int fno=0;
    for (int ni=dl.next(); ni>=0; ni=dl.next()){
      double p = (k>0) ? pt[fno < k ? fno : k-1] : 0.0;
      sal::DeadlineIterator::sal_advance_test_clock(p);
      fno++;
    }
    // emit "surv:a,b,c|drop:x,y,z"
    const auto& s=dl.sal_survivors(); const auto& d=dl.sal_dropped();
    std::string out="surv:";
    for(size_t i=0;i<s.size();++i){ if(i) out+=","; out+=std::to_string(s[i]); }
    out+="|drop:";
    for(size_t i=0;i<d.size();++i){ if(i) out+=","; out+=std::to_string(d[i]); }
    std::printf("%s\n", out.c_str());
  }
  return 0;
}
