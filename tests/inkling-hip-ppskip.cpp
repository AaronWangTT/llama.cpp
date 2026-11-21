// Isolated gfx1103 PP attention oracle. Write raw FP32 for every output, with an independent scalar check.
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void run(ggml_backend_t be, FILE *out, int id, int n, int nq, int ns, bool named=true) {
    constexpr int d=128, h=24, g=8;
    auto *c=ggml_init({2u<<20, nullptr, true});
    auto *q=ggml_new_tensor_4d(c,GGML_TYPE_F32,d,nq,h,ns);
    auto *k=ggml_new_tensor_4d(c,GGML_TYPE_F16,d,n,g,ns);
    auto *v=ggml_new_tensor_4d(c,GGML_TYPE_F16,d,n,g,ns);
    auto *m=ggml_new_tensor_4d(c,GGML_TYPE_F16,n,nq,h,ns);
    auto *si=ggml_new_tensor_1d(c,GGML_TYPE_F32,h);
    ggml_set_name(m,named?"kq_mask_rel_pos-12":"ordinary_bias");
    auto *y=ggml_flash_attn_ext(c,q,k,v,m,1.f,0.f,0.f); ggml_flash_attn_ext_add_sinks(y,si);
    auto *gf=ggml_new_graph(c); ggml_build_forward_expand(gf,y);
    if (!ggml_backend_supports_op(be,y)) throw std::runtime_error("unsupported");
    auto buf=ggml_backend_alloc_ctx_tensors(c,be); if(!buf) throw std::runtime_error("alloc");
    using T=ggml_fp16_t; auto f16=ggml_fp32_to_fp16; auto f32=ggml_fp16_to_fp32;
    std::vector<float> Q(ggml_nelements(q)),S(h),Y(ggml_nelements(y));
    std::vector<T> K(ggml_nelements(k)),V(ggml_nelements(v)),M(ggml_nelements(m));
    for(int hd=0;hd<h;++hd) S[hd]=(hd%3-1)*.25f;
    for(int s=0;s<ns;++s) for(int kg=0;kg<g;++kg) for(int p=0;p<n;++p){
        const int ix=d*(p+n*(kg+g*s)); K[ix]=f16(((p+kg)%5-2)*.125f);
        for(int j=0;j<d;++j) V[ix+j]=f16(((p*7+kg*3+s*5+j*11)%31-15)*.0625f);
    }
    for(int s=0;s<ns;++s) for(int z=0;z<h;++z) for(int t=0;t<nq;++t){
        Q[d*(t+nq*(z+h*s))]=.25f*(1+(t+z+s)%4);
        for(int p=0;p<n;++p){
            const int pos=n-nq+t;
            bool visible= (p<=pos && (id==1 || pos-p<1024)); // SWA/full final and short prefixes
            if(id==2) visible=false; // a real zero numerator and finite sink for every row
            if(id==3) visible=(t%64==(z*11+s*7)%64 && p==64*((t/64+z*3+s*7)%7)+63) ||
                              (t%64==63 && p==n-1); // individual key/query/head/sequence witness, including last valid K
            if(id==4) visible=((p+98+z*9+s*31)%n<313 && (t+z+p)%11!=0); // loop across physical end, with holes
            if(id==5) visible=true;
            if(id==8) visible=(z==7 && s==1 && t==63 && p==63); // cannot skip
            M[p+n*(t+nq*(z+h*s))]=visible?f16(((p+5*t+z+s)%9-4)*.125f):0xfc00;
        }
    }
    ggml_backend_tensor_set(q,Q.data(),0,Q.size()*sizeof(float));ggml_backend_tensor_set(k,K.data(),0,K.size()*2);
    ggml_backend_tensor_set(v,V.data(),0,V.size()*2);ggml_backend_tensor_set(m,M.data(),0,M.size()*2);ggml_backend_tensor_set(si,S.data(),0,S.size()*4);
    if(ggml_backend_graph_compute(be,gf)!=GGML_STATUS_SUCCESS)throw std::runtime_error("graph");
    ggml_backend_tensor_get(y,Y.data(),0,Y.size()*4);
    if(!std::all_of(Y.begin(),Y.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("nonfinite");
    double mx=0;
    for(int s=0;s<ns;++s)for(int z=0;z<h;++z)for(int t=0;t<nq;++t){
        if (id!=3 && id!=8 && t!=0 && t!=nq-1 && t%61!=0)continue;
        double den=std::exp(double(S[z])); std::array<double,3> num{};int ch[]={0,31,127};
        for(int p=0;p<n;++p){T b=M[p+n*(t+nq*(z+h*s))];if(b==0xfc00)continue;
            int ix=d*(p+n*(z/3+g*s)); double w=std::exp(double(f32(b))+Q[d*(t+nq*(z+h*s))]*f32(K[ix])); den+=w;
            for(int j=0;j<3;++j)num[j]+=w*f32(V[ix+ch[j]]);
        }
        for(int j=0;j<3;++j)mx=std::max(mx,std::abs(Y[ch[j]+d*(z+h*(t+nq*s))]-num[j]/den));
    }
    if(mx>.007 || (id==2 && !std::all_of(Y.begin(),Y.end(),[](float x){return x==0.f;})))throw std::runtime_error("analytic reference");
    if(fwrite(Y.data(),sizeof(float),Y.size(),out)!=Y.size())throw std::runtime_error("write");
    printf("case=%d K=%d Q=%d S=%d named=%d oracle_err=%.7g\n",id,n,nq,ns,int(named),mx);ggml_backend_buffer_free(buf);ggml_free(c);
}
int main(int argc,char **argv){
    if(argc!=4)return 2;ggml_backend_load_all_from_path(argv[1]);auto b=ggml_backend_init_by_name("ROCm0",nullptr);if(!b)return 3;
    std::ifstream maps("/proc/self/maps");std::string l;bool seen=false;while(std::getline(maps,l))if(l.find("libggml-hip.so")!=l.npos){if(l.find(std::string(argv[1])+"/libggml-hip.so")==l.npos)return 4;seen=true;}
    if(!seen)return 5;auto *out=fopen(argv[2],"wbx");if(!out)return 6;
    int which=std::stoi(argv[3]); if(which!=2 && which!=5 && which!=8)return 6; run(b,out,which,2048,256,which==8?2:1);
    int rc=fclose(out);ggml_backend_free(b);return rc!=0;
}
