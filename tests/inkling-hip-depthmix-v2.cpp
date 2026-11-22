// Private native fixture. Inject independent state/score words, but let the scheduler
// allocate the activations. A separately requested CONT must still execute on every replay.
#include "ggml-alloc.h"
#include "ggml-inkling-depthmix.h"
#include <rocprofiler-sdk-roctx/roctx.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using words = std::vector<uint32_t>;
static void ck(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static void set(ggml_tensor * x, const words & v) { ggml_backend_tensor_set(x, v.data(), 0, 4*v.size()); }
static words get(ggml_tensor * x) {
    words v(ggml_nelements(x)); ggml_backend_tensor_get(x, v.data(), 0, 4*v.size()); return v;
}

static void test(ggml_backend_t gpu, ggml_backend_t cpu, inkling_depthmix_make_v2_t make, bool enabled, int s, int q) {
    constexpr int d = 2304, t = 256;
    auto * c = ggml_init({2u<<20, nullptr, true});
    auto * refs = ggml_init({2u<<20, nullptr, true});
    ck(c && refs, "contexts");
    auto tensor = [&](int a, int b, int z) { return ggml_new_tensor_3d(c, GGML_TYPE_F32, a,b,z); };
    auto * query = ggml_new_tensor_3d(refs, GGML_TYPE_F32, d,q,1);
    auto * xp = ggml_new_tensor_3d(refs, GGML_TYPE_F32, s,d,t);
    auto * wp = ggml_new_tensor_3d(refs, GGML_TYPE_F32, s,q,t);
    auto * oracle = ggml_mul_mat(refs,xp,wp); // untagged stock MMVF
    auto * rg = ggml_new_graph_custom(refs,32,false); ggml_build_forward_expand(rg,oracle);
    auto buf = ggml_backend_alloc_ctx_tensors(refs,gpu); ck(buf,"reference allocation");
    auto * n = ggml_rms_norm(c,ggml_concat(c,tensor(d,s-1,t),tensor(d,1,t),1),1.e-6f);
    auto * p = ggml_cont(c,ggml_transpose(c,n));
    auto * dot = ggml_mul_mat(c,n,ggml_repeat(c,query,tensor(d,q,t)));
    auto * w = ggml_soft_max_ext(c,dot,nullptr,1.f/48,0);
    auto * ref = ggml_mul_mat(c,p,w);
    ggml_backend_t backends[] = {gpu,cpu};
    auto sched = ggml_backend_sched_new(backends,nullptr,2,128,false,false);
    if (make) {
        ck(!make(c,sched,cpu,ref),"CPU accepted");
        ggml_prec_set_acc(ref,GGML_PREC_F16);
        ck(!make(c,sched,gpu,ref),"non-F32 precision accepted");
        ggml_prec_set_acc(ref,GGML_PREC_UNDEFINED);
        auto * other = tensor(d,s,t); dot->src[0] = other;
        ck(!make(c,sched,gpu,ref),"different score provenance accepted"); dot->src[0] = n;
        ck(!make(c,sched,gpu,ggml_mul_mat(c,ggml_view_tensor(c,p),w)),"view in place of CONT accepted");
    }
    auto * y = make ? make(c,sched,gpu,ref) : nullptr;
    ck(bool(y)==enabled,"unexpected constructor result");
    if (!y) y = ref;
    auto * g = ggml_new_graph_custom(c,64,false); ggml_build_forward_expand(g,y);
    if (enabled) {
        ck(y->op==GGML_OP_OUT_PROD && y->src[0]==n && y->src[1]->op==GGML_OP_TRANSPOSE && y->src[1]->src[0]==w &&
            ggml_backend_sched_get_tensor_backend(sched,y)==gpu,"unexpected operator or backend pin");
        for (int i=0;i<ggml_graph_n_nodes(g);++i) ck(ggml_graph_node(g,i)!=p,"private CONT still reachable");
    }
    ggml_build_forward_expand(g,p); // separate, observed branch
    for (auto * x : {n,w,y,p}) { ggml_set_output(x); ggml_backend_sched_set_tensor_backend(sched,x,gpu); }
    for (int i=0;i<ggml_graph_n_nodes(g);++i) {
        auto * x=ggml_graph_node(g,i); if (x!=p && x!=y) x->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    }
    ck(ggml_backend_sched_alloc_graph(sched,g),"schedule");
    words X(d*s*t), W(s*q*t), P(X.size()); uint32_t seed=1921;
    auto fill = [&](words & v) { for (auto & u : v) { seed=1664525u*seed+1013904223u;
        float f=(int(seed>>8)-8388608)/4194304.f; std::memcpy(&u,&f,4); } };
    for (int k=0;k<4;++k) {
        fill(X); fill(W);
        if (k&1) {
            const uint32_t v[]={0,0x80000000u,1,0x80000001u,0x7f800000,0xff800000u,0x7fc00123,0xff812345u,0x3f800001,0xbf800000};
            for (int i=0;i<t;++i) for (int j=0;j<s;++j) for (int h=0;h<12;++h) X[(i*s+j)*d+h]=v[(i+3*j+h+k)%10];
            for (size_t i=0;i<W.size();++i) W[i]=v[i%7==0?6:(i%3==0?1:8)];
        }
        for (int i=0;i<t;++i) for (int j=0;j<s;++j) for (int h=0;h<d;++h) P[(i*d+h)*s+j]=X[(i*s+j)*d+h];
        set(n,X); set(w,W); set(xp,P); set(wp,W);
        ck(ggml_backend_graph_compute(gpu,rg)==GGML_STATUS_SUCCESS,"stock compute");
        const auto expected=get(oracle);
        auto poison=expected; for (auto & u : poison) u=~u; set(y,poison);
        poison=P; for (auto & u : poison) u=~u; set(p,poison);
        const auto name="raw_"+std::to_string(s)+"_"+std::to_string(q)+"_"+std::to_string(k);
        roctxRangePushA(name.c_str());
        ck(ggml_backend_sched_graph_compute(sched,g)==GGML_STATUS_SUCCESS,"scheduled compute");
        roctxRangePop();
        if (get(y)!=expected || get(p)!=P) throw std::runtime_error(name+": word mismatch");
    }
    ggml_backend_sched_free(sched); ggml_backend_buffer_free(buf); ggml_free(refs); ggml_free(c);
}

int main(int argc,char ** argv) { try {
    ck(argc==3,"LIBDIR on|off|absent"); const std::string mode=argv[2]; ck(mode=="on" || mode=="off" || mode=="absent","mode");
    namespace fs = std::filesystem; const auto lib=fs::canonical(argv[1]);
    ggml_backend_load_all_from_path(lib.c_str());
    auto gpu=ggml_backend_init_by_name("ROCm0",nullptr), cpu=ggml_backend_init_by_name("CPU",nullptr); ck(gpu && cpu,"backends");
    std::ifstream maps("/proc/self/maps"); bool hip=false;
    for (std::string l;std::getline(maps,l);) { auto i=l.find('/'); if (i==l.npos) continue;
        const fs::path path=l.substr(i); const auto n=path.filename().string(); if (n.rfind("libggml",0)==0) {
            ck(fs::canonical(path).parent_path()==lib,"foreign backend"); hip |= n.rfind("libggml-hip.so",0)==0; } }
    ck(hip,"HIP mapping"); auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu));
    auto make=reinterpret_cast<inkling_depthmix_make_v2_t>(ggml_backend_reg_get_proc_address(reg,inkling_depthmix_make_v2_name));
    ck(bool(make)==(mode!="absent"),"unexpected capability export");
    for(int s:{2,4}) for(int q:{1,3,4}) test(gpu,cpu,make,mode=="on",s,q);
    ggml_backend_free(cpu); ggml_backend_free(gpu); return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"depthmix v2: %s\n",e.what()); return 1; } }
