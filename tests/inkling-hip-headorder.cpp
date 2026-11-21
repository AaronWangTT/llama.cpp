// Attention-only synthetic bandwidth fixture. K/V storage follows llama_kv_cache::get_k/get_v
// and its FA permute: [row][KV head][channel] in memory, unlike contiguous [head][row][channel].
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
static void need(bool x,const char *s){if(!x)throw std::runtime_error(s);}
static void run(ggml_backend_t be,int n,int visible,const char*tag,FILE*out){
 constexpr int D=128,H=24,G=8; ggml_context*c=ggml_init({2u<<20,nullptr,true});auto*q=ggml_new_tensor_4d(c,GGML_TYPE_F32,D,1,H,1);
 auto*kr=ggml_new_tensor_3d(c,GGML_TYPE_F16,D,G,n);auto*vr=ggml_new_tensor_3d(c,GGML_TYPE_F16,D,G,n);
 auto*k=ggml_view_4d(c,kr,D,n,G,1,kr->nb[2],kr->nb[1],kr->nb[3],0);auto*v=ggml_view_4d(c,vr,D,n,G,1,vr->nb[2],vr->nb[1],vr->nb[3],0);
 auto*m=ggml_new_tensor_4d(c,GGML_TYPE_F16,n,1,H,1);ggml_set_name(m,"kq_mask_rel_pos-12");auto*si=ggml_new_tensor_1d(c,GGML_TYPE_F32,H);auto*y=ggml_flash_attn_ext(c,q,k,v,m,1/std::sqrt(float(D)),0.f,0.f);ggml_flash_attn_ext_add_sinks(y,si);auto*g=ggml_new_graph(c);ggml_build_forward_expand(g,y);need(ggml_backend_supports_op(be,y),"unsupported");auto*buf=ggml_backend_alloc_ctx_tensors(c,be);need(buf,"device allocation");std::vector<ggml_fp16_t> raw(ggml_nelements(kr)),mask(ggml_nelements(m));uint32_t seed=276411; auto gen=[&](){seed=1664525u*seed+1013904223u;return (int(seed>>20)-2048)*.00048828125f;};for(auto*t:{kr,vr}){for(auto&z:raw)z=ggml_fp32_to_fp16(gen());ggml_backend_tensor_set(t,raw.data(),0,raw.size()*2);}std::vector<float>Q(D*H),S(H),Y(D*H);for(auto&z:Q)z=gen();for(auto&z:S)z=0;ggml_backend_tensor_set(q,Q.data(),0,Q.size()*4);ggml_backend_tensor_set(si,S.data(),0,S.size()*4);
 for(int h=0;h<H;h++)for(int j=0;j<n;j++){bool valid=visible==1024?((j+513)%n)<visible:j<visible;mask[h*n+j]=ggml_fp32_to_fp16(valid?gen():-INFINITY);}ggml_backend_tensor_set(m,mask.data(),0,mask.size()*2);auto step=[&](){need(ggml_backend_graph_compute(be,g)==GGML_STATUS_SUCCESS,"compute");ggml_backend_synchronize(be);}; for(int j=0;j<3;j++)step();int reps=std::getenv("HEAD_PMC")?3:24;std::string label=std::string("headorder_")+tag;roctxRangePushA(label.c_str());auto a=std::chrono::steady_clock::now();for(int j=0;j<reps;j++)step();auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-a).count();roctxRangePop();ggml_backend_tensor_get(y,Y.data(),0,Y.size()*4);need(std::all_of(Y.begin(),Y.end(),[](float f){return std::isfinite(f);}),"nonfinite output");need(std::fwrite(Y.data(),4,Y.size(),out)==Y.size(),"output save");std::printf("%s keys=%d visible=%d reps=%d ms_per_layer=%.7f\n",label.c_str(),n,visible,reps,sec*1000/reps);std::fflush(stdout);ggml_backend_buffer_free(buf);ggml_free(c);
}
int main(int argc,char**argv){try{need(argc==3,"LIB_DIR OUTPUT_FILE");ggml_backend_load_all_from_path(argv[1]);auto*b=ggml_backend_init_by_name("ROCm0",nullptr);need(b,"no ROCm0");std::ifstream maps("/proc/self/maps");std::string l;bool hip=false;while(std::getline(maps,l))if(l.find("/libggml-hip.so")!=l.npos){need(l.find(std::string(argv[1])+"/libggml-hip.so")!=l.npos,"foreign mapped HIP");hip=true;}need(hip,"missing HIP");auto*out=std::fopen(argv[2],"wbx");need(out,"output exists");run(b,768,514,"short",out);run(b,4352,4098,"global4k",out);run(b,4352,1024,"sliding",out);run(b,33024,32770,"global32k",out);need(!std::fclose(out),"close");ggml_backend_free(b);return 0;}catch(const std::exception&e){std::fprintf(stderr,"headorder: %s\n",e.what());return 1;}}
