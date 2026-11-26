#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-kperf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifdef HOST_LAYOUT
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "inkling-hip-situ-layout.inc"
#endif
static void check(bool c,const char * why) { if (!c) { fprintf(stderr,"FAIL %s\n",why);std::exit(1); } }
static int hits;
static void logger(ggml_log_level,const char * s,void *) { if (!strncmp(s,"INKLING_SITU_HIT ",17)) ++hits; fputs(s,stderr); }
enum Case {NORMAL,UP_EARLY,INTERMEDIATE_OUTPUT,FANOUT,SLICE,VIEW_TAP,WIDTH_PAD,BAD_NAME,NEG_ZERO_BIAS,ONE_ULP,OVERLAP};
static void one(ggml_backend_t be,FILE * f,int nt,Case test,bool on) {
    auto c=ggml_init({2u<<20,nullptr,true}); check(c,"ctx");
    const int w=1152, row=2*w+(test==WIDTH_PAD?2:0);
    auto *pack=ggml_new_tensor_3d(c,GGML_TYPE_F32,row,8,nt);ggml_set_input(pack);
    auto *gate=ggml_view_3d(c,pack,w,8,nt,row*4,8*row*4,0);
    auto *up=ggml_view_3d(c,pack,w,8,nt,row*4,8*row*4,(row/2)*4);
    ggml_set_name(gate,"ffn_moe_gate-14"); ggml_set_name(up,"ffn_moe_up-14");
    auto * s=ggml_scale_bias(c,gate,.25f,test==NEG_ZERO_BIAS?-0.f:0.f);
    auto *a=ggml_scale(c,ggml_tanh(c,s),4.f);auto *sig=ggml_sigmoid(c,gate);auto *act=ggml_mul(c,a,sig);
    auto *u=ggml_scale(c,ggml_tanh(c,ggml_scale(c,up,test==ONE_ULP?std::nextafter(1.f/25.f,0.f):1.f/25.f)),25.f);
    auto *out=ggml_mul(c,act,u);ggml_set_name(out,test==BAD_NAME?"other-14":"ffn_moe_situ-14");ggml_set_output(out);
    auto g=ggml_new_graph(c);if(test==UP_EARLY) ggml_build_forward_expand(g,up); ggml_build_forward_expand(g,out);
    ggml_tensor * tap=nullptr;
    if(test==INTERMEDIATE_OUTPUT) tap=a;
    if(test==SLICE) tap=act;
    if(test==VIEW_TAP) tap=up;
    if(test==FANOUT) { tap=ggml_scale(c,s,3.f);ggml_build_forward_expand(g,tap); }
    if(tap && test!=SLICE) ggml_set_output(tap);
    auto first=std::find(g->nodes,g->nodes+g->n_nodes,s)-g->nodes,split=std::find(g->nodes,g->nodes+g->n_nodes,act)-g->nodes+1;
#ifdef HOST_LAYOUT
    auto alloc=ggml_gallocr_new(ggml_backend_cpu_buffer_type());check(ggml_gallocr_alloc_graph(alloc,g),"gallocr");
#else
    auto buf=ggml_backend_alloc_ctx_tensors(c,be);check(buf,"alloc");
#endif
    if(test==OVERLAP) out->data=(char*)pack->data+64;
    int want=(test==NORMAL||test==UP_EARLY) && on;
#ifdef HOST_LAYOUT
    ggml_backend_cuda_context ctx;ggml_cgraph sub=*g;if(test==SLICE) sub.n_nodes=split;
    bool pass=inkling_situ_dispatch(ctx,&sub,first)>0;
    if(!pass && want) fprintf(stderr,"CPU layout T=%d case=%d pack=%p dst=%p\n",nt,test,pack->data,out->data);
    check(pass==bool(want),"matcher against allocated CPU graph");ggml_gallocr_free(alloc);
#else
    std::vector<uint32_t> data(row*8*nt);uint32_t rnd=4231;
    static const uint32_t edge[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x40800000,0xc0800000,0x41c80000,0xc1c80000,0x42b00000,0xc2b00000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0xffc34567,0x7f832145,0xff812abc};
    constexpr int nedge=sizeof(edge)/sizeof(edge[0]);
    for(int i=0;i<int(data.size());++i) {
        rnd=1664525*rnd+1013904223;int j=i%row, e=j%(row/2);
        if(e<nedge*nedge) data[i]=edge[j<row/2?e/nedge:e%nedge]; // every pair, including two different NaN payloads
        else if((j+i/row)%4==0) data[i]=rnd; // includes subnormal, overflow and quiet/signalling NaNs
        else data[i]=(rnd&0x807fffffu)|((118u+((rnd>>23)%16u))<<23);
    }
    ggml_backend_tensor_set(pack,data.data(),0,data.size()*4);
    auto compute=[&](int begin,int end) {auto sub=*g;sub.nodes+=begin;sub.n_nodes=end-begin;sub.uid=0;check(ggml_backend_graph_compute(be,&sub)==GGML_STATUS_SUCCESS,"graph"); ggml_backend_synchronize(be);};
    auto dump=[&](ggml_tensor *t){ data.resize(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,data.data(),0,data.size()*4);check(fwrite(data.data(),4,data.size(),f)==data.size(),"dump");};
    int h=hits;if(test==SLICE){compute(0,split);dump(tap);compute(split,g->n_nodes);}else compute(0,g->n_nodes);
    if(hits-h!=want) fprintf(stderr,"T=%d case=%d want=%d actual=%d\n",nt,test,want,hits-h);
    check(hits-h==want,"dispatch gate");dump(out);if(tap && test!=SLICE) dump(tap);
    ggml_backend_buffer_free(buf);
#endif
    ggml_free(c);
}
int main(int argc,char **argv) {
    bool on=true;ggml_backend_t be=nullptr;FILE *f=nullptr;
#ifdef HOST_LAYOUT
    (void)argc;(void)argv;
#else
    if(argc!=3)return 2;setenv("GGML_KPERF_TRACE","inkling.situ",1);on=ggml_kperf_on(GGML_KPERF_INKLING_SITU);setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);ggml_log_set(logger,nullptr);ggml_backend_load_all_from_path(argv[1]);be=ggml_backend_init_by_name("ROCm0",nullptr);check(be,"device");
    std::ifstream maps("/proc/self/maps");std::string l;bool hip=false;while(getline(maps,l)){auto j=l.find('/');if(j==l.npos)continue;auto p=l.substr(j),name=p.substr(p.rfind('/')+1);if(name.rfind("libggml",0)) continue;check(p.rfind(std::string(argv[1])+"/",0)==0,"foreign library");hip|=name.rfind("libggml-hip.so",0)==0;}check(hip,"HIP mapping");f=fopen(argv[2],"wbx");check(f,"new output");
#endif
    for(int t:{1,128,256}) { one(be,f,t,NORMAL,on);one(be,f,t,UP_EARLY,on); }
    for(auto test:{INTERMEDIATE_OUTPUT,FANOUT,SLICE,VIEW_TAP,WIDTH_PAD,BAD_NAME,NEG_ZERO_BIAS,ONE_ULP,OVERLAP}) one(be,f,1,test,on);
#ifndef HOST_LAYOUT
    check(hits==(on?6:0),"total");check(!fclose(f),"close");ggml_backend_free(be);
#endif
    puts("PASS fixture");
}
