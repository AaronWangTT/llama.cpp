// Regression: one backend, one split key, same node data/shape, different parent consumers.
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-kperf.h"
#include <rocprofiler-sdk-roctx/roctx.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
static void ck(bool x,const char *msg){if(!x){std::fprintf(stderr,"FAIL %s\n",msg);std::exit(1);}}
static int hits=0;static void logger(ggml_log_level,const char*s,void*){if(!std::strncmp(s,"INKLING_SITU_HIT ",17))hits++;std::fputs(s,stderr);}
int main(int ac,char **av){if(ac!=3)return 2;ck(!getenv("GGML_CUDA_DISABLE_GRAPHS"),"graphs must be enabled");setenv("GGML_KPERF_TRACE","inkling.situ",1);bool on=ggml_kperf_on(GGML_KPERF_INKLING_SITU);ggml_log_set(logger,nullptr);ggml_backend_load_all_from_path(av[1]);auto*b=ggml_backend_init_by_name("ROCm0",nullptr);ck(b,"no GPU");std::ifstream maps("/proc/self/maps");std::string l;bool hip=false;while(std::getline(maps,l)){if(l.find("/libggml-hip.so")!=l.npos){ck(l.find(std::string(av[1])+"/libggml-hip.so")!=l.npos,"wrong mapped HIP");hip=true;}}ck(hip,"missing HIP mapping");auto*c=ggml_init({4u<<20,nullptr,true});auto*pack=ggml_new_tensor_3d(c,GGML_TYPE_F32,2304,8,1);ggml_set_input(pack);auto*gate=ggml_view_3d(c,pack,1152,8,1,2304*4,8*2304*4,0);auto*up=ggml_view_3d(c,pack,1152,8,1,2304*4,8*2304*4,1152*4);ggml_set_name(gate,"ffn_moe_gate-14");ggml_set_name(up,"ffn_moe_up-14");auto*s=ggml_scale(c,gate,.25f);auto*tanh_node=ggml_tanh(c,s);auto*ga=ggml_scale(c,tanh_node,4.f);auto*sg=ggml_sigmoid(c,gate);auto*act=ggml_mul(c,ga,sg);auto*ua=ggml_scale(c,ggml_tanh(c,ggml_scale(c,up,1.f/25.f)),25.f);auto*out=ggml_mul(c,act,ua);ggml_set_name(out,"ffn_moe_situ-14");ggml_set_output(out);auto*tap=ggml_scale(c,tanh_node,3.f);ggml_set_output(tap);
 auto*plain=ggml_new_graph(c);auto*tapped=ggml_new_graph(c);ggml_build_forward_expand(plain,out);ggml_build_forward_expand(tapped,out);ggml_build_forward_expand(tapped,tap);int n=plain->n_nodes;ck(tapped->n_nodes==n+1,"wrong tap graph");for(int i=0;i<n;i++)ck(plain->nodes[i]==tapped->nodes[i],"different node identity");auto bp=ggml_backend_alloc_ctx_tensors(c,b);ck(bp,"alloc");auto *f=std::fopen(av[2],"wbx");ck(f,"output");uint32_t rng=46;std::vector<float> data(2304*8),sentinel(1152*8,-99.f),vals(1152*8);auto compute=[&](ggml_cgraph*g,int lo,int hi){ggml_cgraph split=*g;split.nodes=g->nodes+lo;split.n_nodes=hi-lo;split.size=0;split.n_leafs=0;split.leafs=nullptr;split.grads=nullptr;split.grad_accs=nullptr;split.uid=0;ck(ggml_backend_graph_compute(b,&split)==GGML_STATUS_SUCCESS,"compute");ggml_backend_synchronize(b);};auto save=[&](ggml_tensor*x){ggml_backend_tensor_get(x,vals.data(),0,vals.size()*4);ck(std::all_of(vals.begin(),vals.end(),[](float v){return std::isfinite(v);}),"non-finite");ck(std::fwrite(vals.data(),4,vals.size(),f)==vals.size(),"write");};
 for(int phase=0;phase<3;phase++){bool observe=phase==1;
  for(int step=0;step<5;step++){for(auto&v:data){rng=1664525u*rng+1013904223u;v=(int(rng>>16)-32768)/65536.f;}ggml_backend_tensor_set(pack,data.data(),0,data.size()*4);ggml_backend_tensor_set(tanh_node,sentinel.data(),0,sentinel.size()*4);auto marker=std::string("situ_replay_")+std::to_string(phase)+"_"+std::to_string(step);int prior=hits;roctxRangePushA(marker.c_str());compute(observe?tapped:plain,0,n);roctxRangePop();if(observe){compute(tapped,n,n+1);save(tap);ck(std::all_of(vals.begin(),vals.end(),[](float v){return std::abs(v)<=3.01f;}),"stale observed intermediate after graph reuse");}save(out);std::printf("phase=%d step=%d dispatch_matcher_hits=%d\n",phase,step,hits-prior);if(observe)ck(hits==prior,"tapped graph fused");}
 }
 ck(!on||hits>0,"never captured fused candidate");ck(!std::fclose(f),"close");ggml_backend_buffer_free(bp);ggml_free(c);ggml_backend_free(b);}
