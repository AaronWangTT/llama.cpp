#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define INKLING_HOTPAIR_MAP_ONLY
#include "../ggml/src/ggml-cuda/inkling-hotpair.cuh"
struct S {int expert=-1,col_low=-1,col_high=-1;};
static void ck(bool v,const char*m){if(!v)throw std::runtime_error(m);}
static std::pair<int,int> test(const std::array<int,128>&n){std::vector<S>d(192);int p=0,expected=0,full=0;for(int e=0;e<128;e++){int end=p+n[e];expected+=n[e]/64;while(true){int hi=std::min(p+32,end),j=(p+31)/32;if(hi<=j*32){ck(d[64+e].expert<0,"duplicate extra");d[64+e]={e,p,hi};break;}ck(j<64&&d[j].expert<0,"duplicate seam");d[j]={e,p,hi};p=hi;}p=end;}ck(p==2048,"route length");int pairs=0,visits=0;std::array<int,2048>coverage{};for(int i=0;i<192;i++){const auto a=d[i];ck(a.expert>=0,"unfilled descriptor");int r=inkling_hotpair_role(d.data(),i,64);if(a.col_high>a.col_low)visits++;if(a.col_high-a.col_low==32)full++;if(r<0){ck(i>0&&inkling_hotpair_role(d.data(),i-1,64)==1,"orphan follower");continue;}int hi=a.col_high;if(r>0){ck(i+1<64&&inkling_hotpair_role(d.data(),i+1,64)==-1&&a.expert==d[i+1].expert&&a.col_high==d[i+1].col_low&&d[i+1].col_high-a.col_low==64,"unsafe pair");hi=d[i+1].col_high;pairs++;}for(int x=a.col_low;x<hi;x++)coverage.at(x)++;}ck(pairs==expected&&std::all_of(coverage.begin(),coverage.end(),[](int x){return x==1;}),"missing, duplicate or wrong owner");return {pairs,visits};}
int main(int argc,char**argv){try{ck(argc==2,"counts.csv");std::ifstream f(argv[1]);int groups=0,pairs=0,tiles=0;for(std::string l;std::getline(f,l);){std::replace(l.begin(),l.end(),',',' ');std::istringstream in(l);int layer,u;in>>layer>>u;std::array<int,128> n{};for(int &x:n){ck(bool(in>>x)&&x>=0&&x<=256,"invalid count");}auto r=test(n);if(layer<20||layer>=24){groups++;pairs+=r.first;tiles+=r.second;}}ck(groups==704&&pairs==14455&&tiles==70139,"not the recorded route census");
 // Shift every expert start by all 32 seam offsets and test below/at/above a pair or a second pair.
 for(int off=0;off<32;off++)for(int hot:{0,1,31,32,33,63,64,65,95,96,127,128,129,255,256}){std::array<int,128> n{};n[0]=off;n[1]=hot;n[127]=hot;int left=2048-off-2*hot;for(int e=2;left;e++){n[e]=std::min(left,256);left-=n[e];}test(n);}printf("PASS %d IQ4 layer-batches, %d pairs / %d original nonempty descriptors plus 480 boundary distributions\n",groups,pairs,tiles);return 0;}catch(const std::exception&e){fprintf(stderr,"map gate: %s\n",e.what());return 1;}}
