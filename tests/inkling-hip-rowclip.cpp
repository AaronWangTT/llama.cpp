// Inject x immediately after the optional scale/conv; execute ONLY the graph slice
// starting at SQR, with its original use counts. The prefix is structural metadata,
// not a test of Q/R scaling or K convolution. Compare separate OFF/ON processes.
#include "ggml-kperf.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void insist(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static uint32_t bits(float f) { uint32_t v; memcpy(&v,&f,4); return v; }
static int hits, hc, hr, ht;
static void logger(ggml_log_level, const char * msg, void *) {
    if (sscanf(msg,"INKLING_ROWCLIP_HIT cols=%d rows=%d threads=%d",&hc,&hr,&ht)==3) ++hits;
    fputs(msg,stderr);
}
enum Case { MATCH, SAME_ADDR, BAD_ORIGIN, TWO_TOKENS, TAP, FANOUT, EPS, LOWER_NEG_ZERO, REVERSE_DIV, VIEW_OUT };
static void one(ggml_backend_t be, FILE * dump, int cols,int rows,Case test,float threshold,int round,bool on) {
    auto * c=ggml_init({2u<<20,nullptr,true}); insist(c,"ggml_init");
    int tokens=test==TWO_TOKENS?2:1;
    auto * raw=ggml_new_tensor_2d(c,GGML_TYPE_F32,cols*rows,tokens);
    ggml_tensor * x;
    if (rows==8) {
        auto * w=ggml_new_tensor_2d(c,GGML_TYPE_F32,3,cols*rows);
        ggml_set_name(w,test==BAD_ORIGIN?"other.weight":"blk.5.attn_k_conv.weight");
        auto * win=ggml_new_tensor_3d(c,GGML_TYPE_F32,2+tokens,cols*rows,1);
        auto * v=ggml_reshape_2d(c,ggml_ssm_conv(c,win,w),cols*rows,tokens);
        if (round % 2) v=ggml_add(c,v,raw); // optional dynamic lag sum
        auto * scale=ggml_scale(c,ggml_add(c,raw,v),0.5f);
        x=ggml_reshape_3d(c,scale,cols,rows,tokens);
    } else {
        auto * w=ggml_new_tensor_2d(c,GGML_TYPE_F32,cols,rows);
        ggml_set_name(w,test==BAD_ORIGIN?"other.weight":(cols==16?"blk.5.attn_rel_clip.weight":"blk.5.attn_q_clip.weight"));
        x=ggml_mul(c,ggml_reshape_3d(c,raw,cols,rows,tokens),w);
    }
    auto * sq=ggml_sqr(c,x); auto * sum=ggml_sum_rows(c,sq);
    auto * norm=ggml_scale_bias(c,ggml_sqrt(c,sum),1.0f,test==EPS?2e-6f:1e-6f);
    auto * fill=ggml_fill(c,norm,threshold);
    auto * ratio=test==REVERSE_DIV?ggml_div(c,norm,fill):ggml_div(c,fill,norm);
    auto * factor=ggml_clamp(c,ratio,test==LOWER_NEG_ZERO?-0.f:0.f,1.f);
    auto * out=test==VIEW_OUT?ggml_mul_inplace(c,x,factor):ggml_mul(c,x,factor);
    ggml_set_name(out,"Kcur-5"); ggml_set_output(out);
    if(test==TAP) ggml_set_output(norm);
    auto * g=ggml_new_graph(c); ggml_build_forward_expand(g,out);
    auto * tapped=test==TAP?norm:nullptr;
    if(test==FANOUT) { tapped=ggml_scale(c,sum,3.f); ggml_set_output(tapped); ggml_build_forward_expand(g,tapped); }
    auto buf=ggml_backend_alloc_ctx_tensors(c,be); insist(buf,"alloc");
    if(test==SAME_ADDR) out->data=x->data; // exact allocator-style alias without an in-place op
    std::vector<uint32_t> data(cols*rows*tokens);
    for(int h=0;h<rows*tokens;++h) for(int j=0;j<cols;++j) {
        uint32_t rnd=(unsigned(j*31+h+round*17)*2654435761u);
        uint32_t v=(rnd&0x807fffffu)|((115+((rnd>>24)%24))<<23);
        // Cycle the special rows in separately so finite rows still probe reduction order.
        switch ((h+round*5)%15) {
          case 0: v=(j&1)?0u:0x80000000u; break;
          case 1: v=j==0?bits(1.f):0x80000000u; break;
          case 2: v=j==cols-1?bits(-1.f):0u; break;
          case 3: v=j%3?bits(1.00000012f):bits(0.00057f); break;
          case 4: v=j%2?1u:0x80000001u; break;
          case 5: if(j==cols/2) v=0x7fc12345u; break;
          case 6: if(j==0) v=0x7f800000u; break;
          case 7: if(j==2) v=0xff800000u; break;
          case 8: if(j==4) v=0xffa12345u; break;
          case 9: v=j==0?0x7f7fffffu:bits(-0.f); break;
        }
        data[h*cols+j]=v;
    }
    ggml_backend_tensor_set(x,data.data(),0,data.size()*4);
    ggml_cgraph slice=*g; while(slice.n_nodes && *slice.nodes!=sq) { ++slice.nodes; --slice.n_nodes; }
    insist(slice.n_nodes>=8,"missing clip slice"); slice.n_leafs=0; slice.uid=0;
    int before=hits;
    insist(ggml_backend_graph_compute(be,&slice)==GGML_STATUS_SUCCESS,"compute"); ggml_backend_synchronize(be);
    int want=on && (test==MATCH || test==SAME_ADDR);
    insist(hits-before==want,"wrong candidate/fallback dispatch count");
    if(want) insist(hc==cols && hr==rows && ht==(rows==8?512:32),"wrong reduce launch shape");
    for(auto * t:{out,tapped}) if(t) {
        data.resize(ggml_nbytes(t)/4); ggml_backend_tensor_get(t,data.data(),0,data.size()*4);
        insist(fwrite(data.data(),4,data.size(),dump)==data.size(),"write output");
    }
    ggml_backend_buffer_free(buf); ggml_free(c);
}
int main(int argc,char **argv) {
    if(argc!=3) return 2;
    // the OFF arm runs with GGML_KPERF_OFF=inkling.rowclip
    setenv("GGML_KPERF_TRACE","inkling.rowclip",1); setenv("GGML_CUDA_DISABLE_GRAPHS","1",1); bool on=ggml_kperf_on(GGML_KPERF_INKLING_ROWCLIP);
    ggml_log_set(logger,nullptr); ggml_backend_load_all_from_path(argv[1]); auto be=ggml_backend_init_by_name("ROCm0",nullptr); insist(be,"no HIP");
    std::ifstream maps("/proc/self/maps"); std::string line; bool found=false;
    while(getline(maps,line)) if(line.find("libggml-hip.so")!=line.npos) { insist(line.find(std::string(argv[1])+"/libggml-hip.so")!=line.npos,"foreign mapping"); found=true; } insist(found,"missing mapping");
    FILE * f=fopen(argv[2],"wbx"); insist(f,"new dump required");
    const float edge=1.f+1e-6f;
    for(auto shape:{std::pair<int,int>{128,24},{128,8},{16,24}}) {
        int c=shape.first,r=shape.second;
        for(float t:{std::nextafter(edge,0.f),edge,std::nextafter(edge,2.f),4.f,0.f,-0.f,-1.f,INFINITY,NAN}) {
            for(int round=0;round<3;++round) one(be,f,c,r,MATCH,t,round,on);
        }
        for(auto test:{SAME_ADDR,BAD_ORIGIN,TWO_TOKENS,TAP,FANOUT,EPS,LOWER_NEG_ZERO,REVERSE_DIV,VIEW_OUT}) one(be,f,c,r,test,edge,0,on);
    }
    insist(fclose(f)==0,"close"); insist(hits==(on?84:0),"total hits");
    printf("fixture: %d hits; compare OFF/ON files byte-for-byte\n",hits); ggml_backend_free(be);
}
