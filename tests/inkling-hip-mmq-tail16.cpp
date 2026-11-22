// Read one GGUF tensor as packed bytes. Only activations and results are F32.
#include <ggml.h>
#include <gguf.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
static void need(bool ok, const char *msg) { if (!ok) { std::fprintf(stderr,"focused: %s\n",msg); std::exit(1); } }
int main(int argc,char **argv) {
    need(argc==6,"LIBDIR GGUF gate|real|down|q8|ub128|misnamed|wrongpair|decode OUT.f32 REPLAY_IDS");
    const std::string lib=argv[1], mode=argv[3]; const bool down=mode=="down", q8=mode=="q8";
    need(down || q8 || mode=="real" || mode=="gate" || mode=="ub128" || mode=="misnamed" || mode=="wrongpair" || mode=="decode","mode");
    const int k=down ? 1152 : 2304, t=mode=="decode" ? 1 : mode=="ub128" ? 128 : 256, channels=down ? 8 : 1;
    const std::string name="blk."+std::to_string(q8?20:6)+(down?".ffn_down_exps.weight":".ffn_gate_up_exps.weight");
    auto *meta=gguf_init_from_file(argv[2],{true,nullptr}); need(meta,"GGUF");
    auto wi=gguf_find_tensor(meta,name.c_str()); need(wi>=0,"no tensor");
    auto ne=gguf_get_tensor_ne(meta,wi); auto type=gguf_get_tensor_type(meta,wi);
    need(type==(q8?GGML_TYPE_Q8_0:GGML_TYPE_IQ4_NL) && ne[0]==k && ne[1]==2304 && ne[2]==128 && ne[3]==1,"unexpected weight");
    ggml_backend_load_all_from_path(lib.c_str()); auto *be=ggml_backend_init_by_name("ROCm0",nullptr); need(be,"no ROCm0");
    std::ifstream maps("/proc/self/maps"); bool hip=false; std::string line;
    while (std::getline(maps,line)) { auto i=line.find('/'); if(i==line.npos) continue; auto path=line.substr(i); if(path.substr(path.rfind('/')+1).rfind("libggml",0)!=0) continue;
        need(!path.rfind(lib+"/",0),"foreign ggml lib"); hip|=path.find("/libggml-hip.so")!=path.npos; } need(hip,"no mapped HIP");
    auto *c=ggml_init({2u<<20,nullptr,true}); need(c,"context");
    auto *w=ggml_new_tensor_3d(c,type,k,2304,128); ggml_set_name(w,mode=="misnamed" ? "blk.6.ffn_gate_up_exps.weight.extra" : name.c_str());
    auto *x=ggml_new_tensor_3d(c,GGML_TYPE_F32,k,channels,t);
    // Real top-k indices can be a strided view into an expert-width row.
    auto *iraw=ggml_new_tensor_2d(c,GGML_TYPE_I32,128,t); auto *ids=ggml_view_2d(c,iraw,8,t,iraw->nb[1],0);
    auto *y=ggml_mul_mat_id(c,w,x,ids); ggml_set_name(y, down?"ffn_moe_down-24":q8?"ffn_moe_gate_up-20":mode=="wrongpair"?"ffn_moe_gate_up-25":"ffn_moe_gate_up-24");
    auto *g=ggml_new_graph(c); ggml_build_forward_expand(g,y); need(ggml_backend_supports_op(be,y),"op unsupported");
    auto *buf=ggml_backend_alloc_ctx_tensors(c,be); need(buf,"allocation"); need(ggml_nbytes(w)==gguf_get_tensor_size(meta,wi),"weight length");
    std::ifstream file(argv[2],std::ios::binary); file.seekg(gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,wi));
    std::vector<char> bytes(1<<20); for (size_t off=0;off<ggml_nbytes(w);) {size_t n=std::min(bytes.size(),ggml_nbytes(w)-off);
        need(bool(file.read(bytes.data(),n)),"packed read"); ggml_backend_tensor_set(w,bytes.data(),off,n); off+=n; } gguf_free(meta);
    uint32_t seed=451; std::vector<float> X(ggml_nelements(x)), Y(ggml_nelements(y)); std::vector<int32_t> I(128*t,-1);
    auto *out=std::fopen(argv[4],"wbx"); need(out,"output exists or is not writable");
    // Reuse allocations, change Q and routes, then poison and inspect EVERY write.
    std::vector<int> real(256*8); std::ifstream raw(argv[5]); for (int &e: real) { need(bool(raw>>e) && e>=0 && e<128,"bad replay"); } int extra;need(!(raw>>extra),"long replay");
    for(int rep=0;rep<2;rep++) for(int skew=0;skew<((mode=="gate"||mode=="real")?3:2);skew++) {
        for(auto &v:X) {seed=seed*1664525u+1013904223u;v=(int(seed>>16)-32768)/8192.f;}
        for (int row=0;row<t*channels;row++) X[size_t(row)*k]=float(row+1)/512.f;
        ggml_backend_tensor_set(x,X.data(),0,ggml_nbytes(x));
        for(int j=0;j<t;j++) { bool used[128]={}; for(int e=0;e<8;e++) {
            int id=skew?(e<6?(e+j/119)%6:j==t-1 && e==7?127:6+(j*11+(e-6)*13)%119):(5*j+(j<128?17:19)*e)%128;
            if(mode=="real") { id=real[8*((j+7*rep+skew*16)%256)+e]; }
            else if(mode=="gate") {
                if(skew==0) id=8*(j%16)+e; // only 16-column bodies
                if(skew==1) id=8*(j%8)+e;  // only full J32 bodies
                if(skew==2) id=e==0 ? (j<1?0:j<17?1:j<34?2:j<66?3:j<99?4:5) : 6+(7*j+e-1)%122;
                id=(id+19*rep)%128;
            } else id=(id+19*rep)%128; need(!used[id],"duplicate expert in top-k"); used[id]=true; I[128*j+e]=id;
        }}
        ggml_backend_tensor_set(iraw,I.data(),0,ggml_nbytes(iraw));
        std::fill(Y.begin(),Y.end(),std::numeric_limits<float>::quiet_NaN()); ggml_backend_tensor_set(y,Y.data(),0,ggml_nbytes(y));
        need(ggml_backend_graph_compute(be,g)==GGML_STATUS_SUCCESS,"compute"); ggml_backend_synchronize(be);
        ggml_backend_tensor_get(y,Y.data(),0,ggml_nbytes(y)); need(std::all_of(Y.begin(),Y.end(),[](float v){return std::isfinite(v);}),"partial/nonfinite result");
        need(std::fwrite(Y.data(),sizeof(float),Y.size(),out)==Y.size(),"write");
    }
    need(!std::fclose(out),"close");ggml_backend_buffer_free(buf);ggml_free(c);ggml_backend_free(be);
}
