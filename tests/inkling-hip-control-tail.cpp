#include <rocprofiler-sdk-roctx/roctx.h>
// Compare raw F32 bytes to the separate baseline backend. Inject projection
// outputs, not artificial weights that could erase NaN payloads in a matmul.
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-kperf.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
static void ck(bool ok,const char *s) { if (!ok) { fprintf(stderr,"FAIL: %s\n",s); exit(1); } }
static int hits;
static void logger(ggml_log_level,const char *s,void*) { if (!strncmp(s,"INKLING_CTRL_TAIL_HIT ",22)) ++hits; fputs(s,stderr); }
int main(int argc,char **argv) {
    if (argc!=5) return 2; // prefix/lib, T, role, exclusive output file
    const int rows=atoi(argv[2]); const std::string role=argv[3];
    ck((rows==1||rows==256) && (role=="q"||role=="k"||role=="v"||role=="ffn"),"args");
    setenv("GGML_KPERF_TRACE","inkling.ctrl_tail",1); const bool on=ggml_kperf_on(GGML_KPERF_INKLING_CTRL_TAIL); // OFF arm: GGML_KPERF_OFF=inkling.ctrl_tail
    ck(!on||!getenv("GGML_CUDA_DISABLE_GRAPHS"),"positive arm needs replay");
    ggml_log_set(logger,nullptr); ggml_backend_load_all_from_path(argv[1]); auto *b=ggml_backend_init_by_name("ROCm0",nullptr); ck(b,"backend");
    std::ifstream maps("/proc/self/maps"); std::string l; bool hip=false;
    while (std::getline(maps,l)) { auto i=l.find('/'); if(i==l.npos) continue; auto path=l.substr(i); auto name=path.substr(path.rfind('/')+1);
        if(name.rfind("libggml",0)!=0)continue; ck(path.rfind(std::string(argv[1])+"/",0)==0,"wrong ggml path"); hip|=name.rfind("libggml-hip.so",0)==0; }
    ck(hip,"HIP unmapped"); auto *f=fopen(argv[4],"wbx"); ck(f,"output exists");
    auto *c=ggml_init({8u<<20,nullptr,true}); const int n=2304*rows;
    auto dense=[&](int w,int h){return ggml_new_tensor_2d(c,GGML_TYPE_F32,w,h);};
    auto *cur=dense(2304,rows), *msg=dense(2304,rows), *spare=dense(n+1,1);
    auto weight=[&](int w,int h,const char *suffix){auto*t=dense(w,h);ggml_set_name(t,("blk.12.loop_"+role+"_ctrl_"+suffix+".weight").c_str());return t;};
    auto *dw=weight(4608,64,"down"), *fw=weight(64,2304,"forget"), *gw=weight(64,2304,"gain");
    auto *down=ggml_mul_mat(c,dw,ggml_concat(c,ggml_rms_norm(c,cur,1e-6f),ggml_rms_norm(c,msg,1e-6f),0));
    auto *fp=ggml_mul_mat(c,fw,down), *gp=ggml_mul_mat(c,gw,down);
    auto *ft=ggml_tanh(c,fp), *gt=ggml_tanh(c,gp);
    auto *fm=ggml_mul(c,ft,cur), *gm=ggml_mul(c,gt,msg);
    auto *sub=ggml_sub(c,cur,fm), *add=ggml_add(c,sub,gm);
    // Independent sentinel work can be interleaved with either three-node run.
    auto *ind=ggml_dup(c,msg); ggml_set_output(ind); ggml_set_output(add);
    auto *plain=ggml_new_graph(c); ggml_build_forward_expand(plain,add); ggml_build_forward_expand(plain,ind);
    auto *tappedT=ggml_graph_dup(c,plain,false), *tappedM=ggml_graph_dup(c,plain,false);
    for (auto *x:{ft,gt}) ggml_build_forward_expand(tappedT,ggml_dup(c,x));
    for (auto *x:{fm,gm}) ggml_build_forward_expand(tappedM,ggml_dup(c,x));
    auto *after=ggml_add(c,add,cur);ggml_set_output(after);auto *fba=ggml_new_graph(c);ggml_build_forward_expand(fba,after);ggml_build_forward_expand(fba,ggml_dup(c,gt));auto *fbb=ggml_graph_dup(c,fba,false);ggml_build_forward_expand(fbb,ggml_dup(c,add));
    auto *buf=ggml_backend_alloc_ctx_tensors(c,b); ck(buf,"buffer");
    std::vector<uint32_t> data(n),read(n),sent(n,0x7fcdecaf);
    constexpr uint32_t patterns[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f1fffff,0x3f200000,0x3f200001,0x3f7fffff,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f801abc,0xff802def};
    uint32_t seed=7823; auto fill=[&](ggml_tensor*t,int lane,int step){for(int i=0;i<n;++i){seed=seed*1664525u+1013904223u; unsigned j=(i+step*23u); data[i]=i%4==3?seed:patterns[(lane==0?j:lane==1?j/20:lane==2?j/400:j*7)%20];}
        // Fixed cancellation/FMA witness, signed zeros, and competing NaN payloads.
        const uint32_t probes[][4]={{0x3f000000,0x3dcccccd,0x3dcccccd,0xbd3d487f},
            {0x80000000,0x80000000,0x3f800000,0x80000000},
            {0x7fc00011,0xffc00022,0x7f800033,0xff800044},
            {0xff800055,0x7f800066,0xffc00077,0x7fc00088}};
        for(int k=0;k<4;++k)data[k]=probes[k][lane];
        ggml_backend_tensor_set(t,data.data(),0,n*4);};
    auto get=[&](ggml_tensor*t){ggml_backend_tensor_get(t,read.data(),0,n*4);};
    auto save=[&](ggml_tensor*t){get(t);ck(fwrite(read.data(),4,n,f)==(size_t)n,"write");};
    auto split=[&](ggml_cgraph *parent,std::vector<ggml_tensor*> nodes,uint64_t uid){ggml_cgraph g=*parent;g.nodes=nodes.data();g.n_nodes=(int)nodes.size();g.uid=uid;g.n_leafs=0;g.leafs=nullptr;g.size=0;g.grads=nullptr;g.grad_accs=nullptr;ck(ggml_backend_graph_compute(b,&g)==GGML_STATUS_SUCCESS,"compute");ggml_backend_synchronize(b);};
    // Distinct physical split keys (first tensor); same nonzero UID deliberately
    // held across changes to OUTSIDE consumers. Also exercise UID=0 split views.
    for(int side=0;side<2;++side) {
        auto *p=side?gp:fp,*t=side?gt:ft,*m=side?gm:fm,*out=side?add:sub,*w=side?gw:fw;
        const std::string name=w->name; void *oldp=p->data,*oldo=out->data;
        for(int phase=0;phase<11;phase++) {
            // +, external observer of tanh, +, external observer of mul, +,
            // misnamed weight, independent op, exact alias, shifted alias, flagged output, +.
            const bool eligible=(phase==0||phase==2||phase==4||phase==7||phase==10);
            ggml_cgraph *parent=phase==1?tappedT:phase==3?tappedM:plain;
            if(phase==5)ggml_set_name(w,"blk.12.loop_o_ctrl_gain.weight");
            if(phase==7||phase==8){p->data=spare->data;out->data=(char*)spare->data+(phase==8?4:0);}
            if(phase==9)t->flags|=GGML_TENSOR_FLAG_OUTPUT;
            std::vector<ggml_tensor*> nodes={t,m,out}; if(phase==6)nodes.insert(nodes.begin()+1,ind);
            for(int step=0;step<5;++step){
                fill(p,0,phase*5+step);fill(cur,1,step);fill(msg,2,step);if(side)fill(sub,3,step);
                ggml_backend_tensor_set(t,sent.data(),0,n*4);ggml_backend_tensor_set(m,sent.data(),0,n*4);
                if(phase==6)ggml_backend_tensor_set(ind,sent.data(),0,n*4);
                const int prior=hits;std::string tag="ctrl_"+std::to_string(side)+"_"+std::to_string(phase)+"_"+std::to_string(step);roctxRangePushA(tag.c_str());split(parent,nodes,phase<3?100+side:0);roctxRangePop();save(out);
                get(t);ck((read==sent)==(on&&eligible),"wrong tanh materialization/replay");
                get(m);ck((read==sent)==(on&&eligible),"wrong mul materialization/replay");
                if(!eligible){ck(hits==prior,"ineligible fusion");save(t);save(m);}
                if(phase==6){get(msg);auto expect=read;get(ind);ck(expect==read,"independent work skipped");}
            }
            p->data=oldp;out->data=oldo;ggml_set_name(w,name.c_str());if(phase==9)t->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
        }
    }
    add->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    for(int phase=0;phase<3;phase++) for(int step=0;step<5;step++){
        bool observe=phase==1;auto *parent=observe?fbb:fba;fill(gp,0,phase*9+step);fill(cur,1,step);fill(msg,2,step);fill(sub,3,step);
        ggml_backend_tensor_set(add,sent.data(),0,n*4);ggml_backend_tensor_set(gt,sent.data(),0,n*4);
        std::string tag="fallback_add_"+std::to_string(phase)+"_"+std::to_string(step);int prior=hits;
        roctxRangePushA(tag.c_str());split(parent,{gt,gm,add,after},105);roctxRangePop();ck(hits==prior,"observed TANH fused");save(after);save(gt);get(gt);ck(read!=sent,"stale TANH");if(observe){save(add);get(add);ck(read!=sent,"stale ADD after parent gained consumer");}
    }
    ck((hits>0)==on,"wrong hit count");ck(!fclose(f),"close");ggml_backend_buffer_free(buf);ggml_free(c);ggml_backend_free(b);
}
