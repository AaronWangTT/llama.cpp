// Run in the protected GPU window only. Execute the last matmul in isolation so the
// softmax and CONT outputs can contain specific payloads, independent of ROCm libm.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-inkling-depthmix.h"
#include <rocprofiler-sdk-roctx/roctx.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void test(ggml_backend_t be, int s, int q) {
    constexpr int d = 2304, t = 256;
    auto *c = ggml_init({4u<<20, nullptr, true});
    auto tensor = [&](int a, int b, int n) { return ggml_new_tensor_3d(c, GGML_TYPE_F32, a,b,n); };
    auto norm = [&] { return ggml_rms_norm(c, ggml_concat(c,tensor(d,s-1,t),tensor(d,1,t),1),1.e-6f); };
    auto *n=norm(), *other=norm(), *packed=ggml_cont(c,ggml_transpose(c,n));
    auto *packed_view=ggml_view_3d(c,packed,s,d,t,packed->nb[1],packed->nb[2],0);
    auto *rep=ggml_repeat(c,tensor(d,q,1),tensor(d,q,t));
    auto *dot=ggml_mul_mat(c,n,rep), *w=ggml_soft_max_ext(c,dot,nullptr,1.f/48,0.f);
    auto *ref=ggml_mul_mat(c,packed,w), *alt=ggml_mul_mat(c,packed,w);
    auto graph = [&](ggml_tensor *node) { auto *g=ggml_new_graph(c); node->flags |= GGML_TENSOR_FLAG_COMPUTE; ggml_graph_add_node(g,node); return g; };
    auto *ref_g=graph(ref), *alt_g=graph(alt);
    ck(ggml_backend_supports_op(be,alt),"unsupported fixture");
    auto *buf=ggml_backend_alloc_ctx_tensors(c,be); ck(buf,"alloc");
    std::vector<uint32_t> xv(d*s*t), wv(s*q*t), expected(d*q*t), actual(expected.size());
    auto set = [](ggml_tensor *x,const auto &v) { ggml_backend_tensor_set(x,v.data(),0,4*v.size()); };
    // Normalized tensor is not the storage the candidate is allowed to read.
    set(n,xv);
    uint32_t seed=1921;
    auto fill = [&](auto &v) { for (auto &u:v) { seed=seed*1664525u+1013904223u; float f=(int(seed>>8)-8388608)/4194304.f; std::memcpy(&u,&f,4); } };
    for(int k=0;k<2;++k) {
        fill(xv); fill(wv);
        if(k) {
            const uint32_t bits[]={0,0x80000000u,1,0x80000001u,0x7f800000,0xff800000u,0x7fc00123,0xff812345u,0x3f800001,0xbf800000};
            for(int i=0;i<t;++i) for(int j=0;j<s;++j) for(int h=0;h<12;++h) xv[(i*d+h)*s+j]=bits[(i+3*j+h)%10];
            for(size_t i=0;i<wv.size();++i) wv[i]=bits[i%7==0?6:(i%3==0?1:8)];
        }
        set(packed,xv); set(w,wv);
        ck(ggml_backend_graph_compute(be,ref_g)==GGML_STATUS_SUCCESS,"stock");
        ggml_backend_tensor_get(ref,expected.data(),0,expected.size()*4);
        for(int mode=0;mode<5;++mode) {
            // Invert every reference word: a skipped store cannot pass via stale output.
            for(size_t j=0;j<actual.size();++j) actual[j]=~expected[j];
            set(alt,actual);
            alt->src[0]=mode==4?packed_view:packed;
            alt->op_params[inkling_depthmix_slot]=mode==1?0:inkling_depthmix_tag;
            dot->src[0]=mode==2?other:n;
            ggml_prec_set_src(alt, mode==3?GGML_PREC_F16:GGML_PREC_UNDEFINED,1);
            const std::string tag="case_"+std::to_string(s)+"_"+std::to_string(q)+"_"+std::to_string(k)+"_"+std::to_string(mode);
            roctxRangePushA(tag.c_str());
            ck(ggml_backend_graph_compute(be,alt_g)==GGML_STATUS_SUCCESS,"candidate/fallback");
            roctxRangePop();
            ggml_backend_tensor_get(alt,actual.data(),0,actual.size()*4);
            if(actual!=expected) throw std::runtime_error("word mismatch s/q/case/mode="+std::to_string(s)+"/"+std::to_string(q)+"/"+std::to_string(k)+"/"+std::to_string(mode));
        }
        dot->src[0]=n;
    }
    ggml_backend_buffer_free(buf); ggml_free(c);
    std::printf("PASS S=%d Q=%d (%zu words per comparison)\n",s,q,actual.size());
}
int main(int argc,char **argv) { try {
    ck(argc==2 && inkling_depthmix_enabled(),"need LIBDIR and opt-in");
    ggml_backend_load_all_from_path(argv[1]); auto be=ggml_backend_init_by_name("ROCm0",nullptr); ck(be,"ROCm0");
    std::ifstream maps("/proc/self/maps"); bool hip=false;
    for(std::string l;std::getline(maps,l);) { auto i=l.find('/'); if(i==l.npos)continue; auto path=l.substr(i),name=path.substr(path.rfind('/')+1); if(name.rfind("libggml",0)==0) { ck(path.rfind(std::string(argv[1])+"/",0)==0,"foreign library"); hip|=name.rfind("libggml-hip.so",0)==0; } }
    ck(hip,"no HIP mapping"); for(int s:{2,4}) for(int q:{1,3,4}) test(be,s,q); ggml_backend_free(be); return 0;
} catch(const std::exception &e) { std::fprintf(stderr,"fixture: %s\n",e.what()); return 1; } }
